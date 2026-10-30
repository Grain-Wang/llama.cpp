#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from safetensors import safe_open


def tensor_module(name: str) -> str:
    if name.startswith("action_in_proj.") or name.startswith("action_out_proj."):
        return "action_projection"
    if name.startswith("time_mlp_in.") or name.startswith("time_mlp_out."):
        return "time_embedding"
    if name.startswith("paligemma_with_expert.gemma_expert."):
        return "action_expert"
    if ".paligemma.model.vision_tower." in name:
        return "vision_encoder"
    if ".paligemma.model.multi_modal_projector." in name:
        return "multimodal_projector"
    if ".paligemma.model.language_model." in name or name.endswith(".paligemma.lm_head.weight"):
        return "language_backbone"
    raise ValueError(f"unknown tensor module: {name}")


def parameter_count(shape: list[int]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a pi0.5 Safetensors metadata inventory.")
    parser.add_argument("model", type=Path, help="Path to model.safetensors")
    parser.add_argument("output", type=Path, help="Output CSV path")
    args = parser.parse_args()

    rows = []
    with safe_open(args.model, framework="pt", device="cpu") as model:
        for name in model.keys():
            tensor = model.get_slice(name)
            shape = tensor.get_shape()
            rows.append(
                {
                    "source_name": name,
                    "module": tensor_module(name),
                    "dtype": str(tensor.get_dtype()),
                    "shape": "x".join(str(dimension) for dimension in shape),
                    "parameters": parameter_count(shape),
                }
            )

    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["source_name", "module", "dtype", "shape", "parameters"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} tensors to {args.output}")


if __name__ == "__main__":
    main()
