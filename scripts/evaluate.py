from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"n": len(y_true), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary land-use conflict predictions.")
    parser.add_argument("--predictions", required=True, help="CSV with sample_id, label, and predicted_label or probability")
    parser.add_argument("--output", required=True, help="Output JSON metrics path")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    if "label" not in df.columns:
        raise ValueError("Prediction file must contain a label column.")
    if "predicted_label" in df.columns:
        y_pred = df["predicted_label"].astype(int).tolist()
    elif "probability" in df.columns:
        y_pred = (df["probability"].astype(float) >= args.threshold).astype(int).tolist()
    else:
        raise ValueError("Prediction file must contain predicted_label or probability.")

    metrics = binary_metrics(df["label"].astype(int).tolist(), y_pred)
    metrics["threshold"] = args.threshold
    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
