# Land-use conflict identification materials

This repository provides the released data materials and executable scripts for inspecting the data-processing, feature-extraction, and evaluation workflow used in the land-use conflict identification study.

## Repository structure

- `data/raw/`: released source tables and original text files.
- `data/unified_land_use_dataset.csv`: unified dataset table with stable sample identifiers.
- `splits/`: held-out split identifiers and repeated stratified K-fold identifiers.
- `scripts/`: executable preprocessing, feature-extraction, and evaluation scripts.
- `configs/default.yaml`: documented split, feature, and evaluation settings.
- `figures/`: released dataset-summary and feature-analysis figures.
- `results/`: released analysis report.
- `DATASET.md`: dataset fields, split-generation notes, and data-use information.

## Installation

Create a Python environment and install the required package:

```bash
pip install -r requirements.txt
```

The scripts were written for Python 3.9 or later.

## Rebuild the organized dataset and split files

```bash
python scripts/preprocess_dataset.py --input data/raw/land_use_dataset.csv --output-dir .
```

This command regenerates:

- `data/unified_land_use_dataset.csv`
- `splits/heldout_split_ids.csv`
- `splits/repeated_stratified_kfold_ids.csv`

## Extract features

```bash
python scripts/extract_features.py --input data/unified_land_use_dataset.csv --output data/text_features_rebuilt.csv
```

## Evaluate predictions

The evaluation script expects a CSV file containing `sample_id`, `label`, and either `predicted_label` or `probability`.

```bash
python scripts/evaluate.py --predictions predictions.csv --output results/metrics.json
```

If `probability` is supplied, predictions are generated with a default threshold of `0.5` unless another threshold is specified.

## One-step data workflow

```bash
python scripts/run_reproducibility_pipeline.py
```

This convenience command rebuilds the unified dataset, split identifiers, and feature table using the released files in this repository.
