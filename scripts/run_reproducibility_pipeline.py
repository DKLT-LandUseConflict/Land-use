from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    python = sys.executable
    run(
        [
            python,
            "scripts/preprocess_dataset.py",
            "--input",
            "data/raw/land_use_dataset.csv",
            "--output-dir",
            ".",
        ]
    )
    run(
        [
            python,
            "scripts/extract_features.py",
            "--input",
            "data/unified_land_use_dataset.csv",
            "--output",
            "data/text_features_rebuilt.csv",
        ]
    )


if __name__ == "__main__":
    main()
