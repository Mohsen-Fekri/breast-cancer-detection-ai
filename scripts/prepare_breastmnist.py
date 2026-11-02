"""Export BreastMNIST (MedMNIST v2, 224 px) to the negative/ and positive/ folder layout train.py expects.

Download breastmnist_224.npz from https://zenodo.org/records/10519652 (CC BY 4.0).
BreastMNIST label 0 is malignant and label 1 is normal or benign, so malignant becomes positive/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def export(npz_path: Path, output: Path) -> dict[str, dict[str, int]]:
    data = np.load(npz_path, allow_pickle=False)
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        images, labels = data[f"{split}_images"], data[f"{split}_labels"].ravel()
        counts[split] = {"negative": 0, "positive": 0}
        for index, (image, label) in enumerate(zip(images, labels)):
            folder = "positive" if label == 0 else "negative"
            destination = output / split / folder
            destination.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(destination / f"{split}_{index:04d}.png")
            counts[split][folder] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/breastmnist"))
    args = parser.parse_args()
    for split, count in export(args.npz, args.output).items():
        print(f"{split:5s} negative={count['negative']:4d} positive={count['positive']:4d}")


if __name__ == "__main__":
    main()
