from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def stable_sample_ids(df: pd.DataFrame) -> list[str]:
    ids = []
    counters = defaultdict(int)
    for _, row in df.iterrows():
        category = str(row["category"]).strip().lower().replace(" ", "_")
        label = int(row["label"])
        prefix = {"plan_texts": "PLAN", "policy_files": "POLICY", "conflict_cases": "CONFLICT"}.get(category, "SAMPLE")
        counters[prefix] += 1
        ids.append(f"{prefix}_{label}_{counters[prefix]:04d}")
    return ids


def stratified_heldout_split(df: pd.DataFrame, seed: int, test_fraction: float) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for label, group in df.groupby("label", sort=True):
        ids = group["sample_id"].tolist()
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * test_fraction))
        test_ids = set(ids[:n_test])
        for sample_id in ids:
            rows.append({"sample_id": sample_id, "label": int(label), "split": "test" if sample_id in test_ids else "train", "random_seed": seed})
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def repeated_stratified_folds(df: pd.DataFrame, seed: int, n_splits: int, n_repeats: int) -> pd.DataFrame:
    rows = []
    for repeat in range(1, n_repeats + 1):
        rng = random.Random(seed + repeat - 1)
        for label, group in df.groupby("label", sort=True):
            ids = group["sample_id"].tolist()
            rng.shuffle(ids)
            for idx, sample_id in enumerate(ids):
                rows.append({"sample_id": sample_id, "label": int(label), "repeat": repeat, "fold": (idx % n_splits) + 1, "random_seed": seed + repeat - 1})
    return pd.DataFrame(rows).sort_values(["repeat", "fold", "sample_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified dataset and split IDs from the released land-use dataset.")
    parser.add_argument("--input", required=True, help="Path to land_use_dataset.csv")
    parser.add_argument("--output-dir", default=".", help="Output root directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.output_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = ["filename", "title", "content", "source", "category", "label", "word_count"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.loc[:, required].copy()
    df.insert(0, "sample_id", stable_sample_ids(df))
    df["label"] = df["label"].astype(int)
    df["word_count"] = df["word_count"].astype(int)

    df.to_csv(out / "data" / "unified_land_use_dataset.csv", index=False, encoding="utf-8-sig")
    stratified_heldout_split(df, args.seed, args.test_fraction).to_csv(out / "splits" / "heldout_split_ids.csv", index=False, encoding="utf-8-sig")
    repeated_stratified_folds(df, args.seed, args.n_splits, args.n_repeats).to_csv(out / "splits" / "repeated_stratified_kfold_ids.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
