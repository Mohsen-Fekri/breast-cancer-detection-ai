"""Export the checkpoint to ONNX for the in-browser demo served from docs/ by GitHub Pages.

The exported graph returns the logit and a 7x7 Grad-CAM map. With global average pooling followed by one
linear layer, the Grad-CAM channel weights equal the head weights divided by the map area, so the map can be
computed in the forward pass and the browser never needs gradients.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from meddetect.config import DEFAULT_CHECKPOINT, INPUT_SIZE
from meddetect.model import load_checkpoint


class LogitAndCam(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.features = model.features
        self.linear = model.classifier[1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activations = self.features(x)
        logit = self.linear(activations.mean(dim=(2, 3)))
        weights = self.linear.weight[0] / (activations.shape[2] * activations.shape[3])
        cam = torch.relu((weights[None, :, None, None] * activations).sum(dim=1))
        return logit, cam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=Path("docs/model.onnx"))
    args = parser.parse_args()
    model, _ = load_checkpoint(args.checkpoint)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        LogitAndCam(model),
        torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE),
        str(args.output),
        input_names=["image"],
        output_names=["logit", "cam"],
        opset_version=17,
    )
    print(f"saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
