import argparse
import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def load_features(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    data = torch.load(path, map_location='cpu')
    if isinstance(data, dict):
        for key in ['features', 'feature', 'feat', 'feats']:
            if key in data:
                data = data[key]
                break
    return torch.as_tensor(data)


class PhaseDataset(Dataset):
    def __init__(self, annotation_csv: str, feature_dir: str, split: str = 'train', limit_videos=None):
        df = pd.read_csv(annotation_csv)
        df = df[df['split'] == split]
        if limit_videos is not None:
            df = df[df['video_id'].isin(limit_videos)]
        self.samples = []
        for vid, group in df.groupby('video_id'):
            feat_path = os.path.join(feature_dir, f'{vid}.pt')
            feats = load_features(feat_path)
            duration = group['end'].max()
            fps = len(feats) / max(duration, 1e-5)
            for _, row in group.iterrows():
                s = int(row['start'] * fps)
                e = max(int(row['end'] * fps), s + 1)
                e = min(e, len(feats))
                seg_feat = feats[s:e].mean(dim=0)
                self.samples.append((seg_feat, int(row['phase_id'])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train(dataset: PhaseDataset, num_classes: int, epochs: int = 5, lr: float = 1e-3, batch_size: int = 32) -> nn.Module:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = dataset[0][0].shape[0]
    model = nn.Sequential(nn.Linear(input_dim, num_classes))
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for feats, labels in loader:
            optim.zero_grad()
            out = model(feats.float())
            loss = criterion(out, labels.long())
            loss.backward()
            optim.step()
    return model


def predict(model: nn.Module, csv_path: str, feature_dir: str, output_csv: str):
    df = pd.read_csv(csv_path)
    df['Predict_phase_id'] = -1
    for vid, group in df.groupby('Video_name'):
        feat_path = os.path.join(feature_dir, f'{vid}.pt')
        feats = load_features(feat_path)
        frame_count = len(group)
        fps = len(feats) / max(frame_count, 1)
        for idx in group.index:
            fname = df.loc[idx, 'Frame_id']
            frame_idx = int(os.path.splitext(fname)[0].split('_')[-1])
            feat_idx = min(int(frame_idx * fps), len(feats) - 1)
            feat = feats[feat_idx].float().unsqueeze(0)
            with torch.no_grad():
                pred = model(feat).argmax(dim=1).item()
            df.loc[idx, 'Predict_phase_id'] = pred
    df.to_csv(output_csv, index=False)
    print(f'Saved predictions to {output_csv}')


def main():
    parser = argparse.ArgumentParser(description='Train on features and predict phases')
    parser.add_argument('--annotation', required=True, help='annotation csv path')
    parser.add_argument('--features', required=True, help='directory with feature .pt files')
    parser.add_argument('--val2', required=True, help='csv file listing validation frames')
    parser.add_argument('--output', default='pred_val2.csv', help='output csv file')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true', help='train only on the seven sample videos')
    args = parser.parse_args()

    subset = None
    if args.dry_run:
        subset = ['case_0985', 'case_0791', 'case_1362', 'case_1475', 'case_1944', 'case_0690', 'case_0612']

    dataset = PhaseDataset(args.annotation, args.features, split='train', limit_videos=subset)
    num_classes = 35
    model = train(dataset, num_classes=num_classes, epochs=args.epochs)
    predict(model, args.val2, args.features, args.output)


if __name__ == '__main__':
    main()
