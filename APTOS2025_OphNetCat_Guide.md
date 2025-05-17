# APTOS2025 OphNet-Cat Usage Guide

This document summarizes how to obtain the APTOS2025 OphNet-Cat dataset and how to run the baseline experiment.

## 1. Downloading the Dataset

The cataract subset of the OphNet dataset is hosted on [HuggingFace](https://huggingface.co/datasets/xioamiyh/APTOS2025_OphNet-Cat). You can also download from the OneDrive mirror if preferred:
<https://asiateleophth-my.sharepoint.com/:f:/g/personal/secretariat_asiateleophth_org/EosodiUKJjJDgVnDlbKlu2UB0Lkh1gMOkQPeulvF7DlOxA?e=AILLxk>

Use the HuggingFace CLI to download when needed:
```bash
huggingface-cli download --repo-type dataset xioamiyh/APTOS2025_OphNet-Cat --local-dir ./APTOS2025_OphNet-Cat
```

If you already have the data on Google Drive, the raw training videos reside in
`/content/drive/MyDrive/kaggle/APTOS/aptos_videos` (e.g.
`case_0012.mp4`).  The same content is also available as split archives
`/content/drive/MyDrive/kaggle/APTOS/APTOS_train-val/aptos_ophnet.tar.gz.00`
to `aptos_ophnet.tar.gz.24`.  Pre-computed VideoMAE-base features are stored in
`/content/drive/MyDrive/kaggle/APTOS/features_vmae224_b/` with names like
`case_0012.pt`.

Phase annotations for both training and validation are provided in
`/content/drive/MyDrive/kaggle/APTOS/APTOS_train-val_annotation.csv`.  Training
and validation videos share the same folders, so use the `split` column in this
CSV to separate them.  For a quick dry run you can train using only the seven
videos `case_0985`, `case_0791`, `case_1362`, `case_1475`, `case_1944`,
`case_0690` and `case_0612`.

The file `aptos_val2.csv` containing frame names for prediction is located at
`/content/drive/MyDrive/kaggle/APTOS/APTOS_val2.csv`.  The corresponding test
frames are under `/content/drive/MyDrive/kaggle/APTOS/val2_videos/aptos_val2/frames/`.

## 2. Baseline Code

The reference implementation is available at [APTOS2025_OphNet](https://github.com/minghu0830/APTOS2025_OphNet).
Clone the repository and install the required packages:
```bash
git clone https://github.com/minghu0830/APTOS2025_OphNet.git
cd APTOS2025_OphNet
pip install -r requirements.txt
```

## 3. Training

1. The baseline repository expects the dataset in `APTOS2025_OphNet/dataset`.
   If your data lives elsewhere (e.g. on Google Drive), create a symbolic link or
   pass the absolute path when launching `train.py`:
   ```bash
   ln -s /content/drive/MyDrive/kaggle/APTOS APTOS2025_OphNet/dataset
   # or
   python train.py --cfg configs/cataract_phase.yaml \
       --data /content/drive/MyDrive/kaggle/APTOS
   ```
   In `configs/cataract_phase.yaml` set the paths explicitly if needed:
   ```yaml
   data_root: /content/drive/MyDrive/kaggle/APTOS
   annotation_file: /content/drive/MyDrive/kaggle/APTOS/APTOS_train-val_annotation.csv
   feature_dir: /content/drive/MyDrive/kaggle/APTOS/features_vmae224_b
   ```


## 4. Quick Feature Training

The script `train_features.py` trains a simple classifier using the pre-extracted VideoMAE features stored on Google Drive and directly generates predictions for `APTOS_val2.csv`.

```bash
python train_features.py \
  --annotation /content/drive/MyDrive/kaggle/APTOS/APTOS_train-val_annotation.csv \
  --features /content/drive/MyDrive/kaggle/APTOS/features_vmae224_b \
  --val2 /content/drive/MyDrive/kaggle/APTOS/APTOS_val2.csv \
  --output pred_val2.csv --dry-run
```

Use the `--dry-run` flag to train only on the seven example videos listed above. Omit it to use the full training set. The resulting `pred_val2.csv` already contains the `Predict_phase_id` column for submission.
## 5. Inference and Submission

After training, run inference on the validation set:
```bash
python infer.py --cfg configs/cataract_phase.yaml --ckpt <path-to-checkpoint>
```
Append a new column named `Predict_phase_id` to `aptos_val2.csv` and fill in the predicted phase ID for each frame. Submit this CSV file.

## 6. Evaluation Metrics

The official metrics are:
- **Video-level Accuracy**: average frame accuracy per video.
- **Macro Precision**, **Recall**, and **F1 Score** across all phases.
- **Rank Score** = (Video-level Accuracy + Macro F1 Score) / 2.

See the challenge description for full details of how TP, FP and FN are computed.

