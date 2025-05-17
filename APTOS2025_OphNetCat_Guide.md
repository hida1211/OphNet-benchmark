# APTOS2025 OphNet-Cat Usage Guide

This document summarizes how to obtain the APTOS2025 OphNet-Cat dataset and how to run the baseline experiment.

## 1. Downloading the Dataset

The cataract subset of the OphNet dataset is hosted on [HuggingFace](https://huggingface.co/datasets/xioamiyh/APTOS2025_OphNet-Cat). Due to its large size you can also download from the OneDrive mirror:
<https://asiateleophth-my.sharepoint.com/:f:/g/personal/secretariat_asiateleophth_org/EosodiUKJjJDgVnDlbKlu2UB0Lkh1gMOkQPeulvF7DlOxA?e=AILLxk>

Use the HuggingFace CLI to download:
```bash
huggingface-cli download --repo-type dataset xioamiyh/APTOS2025_OphNet-Cat --local-dir ./APTOS2025_OphNet-Cat
```
The provided `aptos_val2.csv` file contains 44,895 frames used for validation.

## 2. Baseline Code

The reference implementation is available at [APTOS2025_OphNet](https://github.com/minghu0830/APTOS2025_OphNet).
Clone the repository and install the required packages:
```bash
git clone https://github.com/minghu0830/APTOS2025_OphNet.git
cd APTOS2025_OphNet
pip install -r requirements.txt
```

## 3. Training

1. Place the downloaded dataset under `APTOS2025_OphNet/dataset`.
2. Follow the configuration in the repository to start training. Example commands:
```bash
python train.py --cfg configs/cataract_phase.yaml --data ./dataset
```
Adjust the configuration file if your dataset path differs.

## 4. Inference and Submission

After training, run inference on the validation set:
```bash
python infer.py --cfg configs/cataract_phase.yaml --ckpt <path-to-checkpoint>
```
Append a new column named `Predict_phase_id` to `aptos_val2.csv` and fill in the predicted phase ID for each frame. Submit this CSV file.

## 5. Evaluation Metrics

The official metrics are:
- **Video-level Accuracy**: average frame accuracy per video.
- **Macro Precision**, **Recall**, and **F1 Score** across all phases.
- **Rank Score** = (Video-level Accuracy + Macro F1 Score) / 2.

See the challenge description for full details of how TP, FP and FN are computed.

