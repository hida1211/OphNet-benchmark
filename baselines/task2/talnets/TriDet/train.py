# python imports
import argparse
import os
import time
import datetime
from pprint import pprint
import shutil
import glob

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (
    train_one_epoch,
    valid_one_epoch,
    ANETdetection,
    save_checkpoint,
    make_optimizer,
    make_scheduler,
    fix_random_seed,
    ModelEma,
)


def prepare_data(dataset_root, feat_src=None, ann_csv=None):
    """Copy features and convert annotations if provided."""
    tridet_dir = os.path.dirname(os.path.abspath(__file__))

    # prepare features
    if feat_src:
        dest_feat = os.path.join(dataset_root, "features", "videomae")
        os.makedirs(dest_feat, exist_ok=True)
        for fname in glob.glob(os.path.join(feat_src, "*.pt")):
            dpath = os.path.join(dest_feat, os.path.basename(fname))
            if not os.path.exists(dpath):
                shutil.copy2(fname, dpath)

    # prepare annotation json
    if ann_csv:
        dest_json = os.path.join(
            dataset_root, "tal_annotations", "OphNet2024_phase.json"
        )
        if not os.path.exists(dest_json):
            import sys

            repo_root = os.path.abspath(
                os.path.join(tridet_dir, "..", "..", "..", "..")
            )
            if repo_root not in sys.path:
                sys.path.append(repo_root)
            from data_processing.csv_to_tridet_json import convert

            os.makedirs(os.path.dirname(dest_json), exist_ok=True)
            convert(ann_csv, dest_json, default_fps=30)


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # resolve dataset root
    tridet_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(tridet_dir, "..", "..", "dataset"))
    dataset_root = os.path.abspath(args.dataset_root or default_root)

    # ensure dataset is available
    prepare_data(dataset_root, args.feat_src, args.anno_csv)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(
                                        cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    best_mAP=0.0
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            print_freq=args.print_freq
        )

        #test for one epoch
        # set up evaluator
        det_eval, output_file = None, None
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
        )

        mAP = valid_one_epoch(
            val_loader,
            model,
            epoch,
            evaluator=det_eval,
            output_file=output_file,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )
        if mAP > best_mAP:
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                True,
                file_folder=ckpt_folder,
                file_name='best.pth.tar'
            )
            print("Best model saved at epoch{}".format(epoch))
            best_mAP=mAP
        print('Current mAP:{}, Best mAP:{}'.format(mAP, best_mAP))

        # save ckpt once in a while
        if (
                (epoch == max_epochs - 1) or
                (
                        (args.ckpt_freq > 0) and
                        (epoch % args.ckpt_freq == 0) and
                        (epoch > 0)
                )
        ):
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )

    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('--config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--dataset-root', default=None, type=str,
                        help='root directory for dataset (default: dataset/)')
    parser.add_argument('--feat-src', default=None, type=str,
                        help='folder containing feature .pt files')
    parser.add_argument('--anno-csv', default=None, type=str,
                        help='annotation csv file to convert')
    args = parser.parse_args()
    main(args)
