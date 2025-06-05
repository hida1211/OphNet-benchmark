import argparse
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def main():
    parser = argparse.ArgumentParser(description="Evaluate phase predictions")
    parser.add_argument("--annotation", required=True, help="CSV with ground truth phase labels")
    parser.add_argument("--pred", required=True, help="CSV with predicted phase labels")
    parser.add_argument("--out", default="metrics.json", help="File to save computed metrics")
    args = parser.parse_args()

    ann = pd.read_csv(args.annotation)
    pred = pd.read_csv(args.pred)

    video_col = "Video_name" if "Video_name" in ann.columns else "video_id"
    frame_col = "Frame_id" if "Frame_id" in ann.columns else "frame"
    label_col = "Phase_id" if "Phase_id" in ann.columns else "phase_id"
    pred_col = "Predict_phase_id" if "Predict_phase_id" in pred.columns else "predict_phase_id"

    df = ann[[video_col, frame_col, label_col]].merge(
        pred[[video_col, frame_col, pred_col]],
        on=[video_col, frame_col],
        how="inner",
    )

    acc_per_video = df.groupby(video_col).apply(
        lambda g: (g[label_col] == g[pred_col]).mean()
    )
    video_accuracy = acc_per_video.mean() if len(acc_per_video) else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        df[label_col],
        df[pred_col],
        labels=list(range(35)),
        average="macro",
        zero_division=0,
    )

    rank_score = (video_accuracy + f1) / 2

    metrics = {
        "video_accuracy": video_accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "rank_score": rank_score,
    }

    print(f"Rank Score: {rank_score:.4f}")
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
