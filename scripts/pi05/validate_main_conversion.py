#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402

from conversion import ModelBase, ModelType, get_model_architecture, get_model_class  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the PI0.5 main-GGUF tensor mapping without writing weights")
    parser.add_argument("model", type=Path, help="directory containing config.json and model.safetensors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hparams = ModelBase.load_hparams(args.model, False)
    architecture = get_model_architecture(hparams, ModelType.TEXT)
    model_class = get_model_class(architecture)
    model = model_class(
        args.model,
        gguf.LlamaFileType.MOSTLY_F16,
        Path("pi05-main-f16.gguf"),
        dry_run=True,
        hparams=hparams,
    )

    mapped: list[str] = []
    skipped: list[str] = []
    for source_name, tensor_loader in model.model_tensors.items():
        outputs = list(model.modify_tensors(tensor_loader(), source_name, None))
        if outputs:
            mapped.extend(target_name for target_name, _ in outputs)
        else:
            skipped.append(source_name)

    if len(mapped) != 371:
        raise ValueError(f"Expected 371 mapped tensors, got {len(mapped)}")
    if len(set(mapped)) != len(mapped):
        raise ValueError("Two or more source tensors map to the same target tensor")
    if len(skipped) != 441:
        raise ValueError(f"Expected 441 skipped tensors, got {len(skipped)}")

    vision_model_class = get_model_class(architecture, mmproj=True)
    vision_model = vision_model_class(
        args.model,
        gguf.LlamaFileType.MOSTLY_F16,
        Path("mmproj-pi05-f16.gguf"),
        dry_run=True,
        hparams=hparams,
    )
    vision_mapped = [
        target_name
        for source_name, tensor_loader in vision_model.model_tensors.items()
        for target_name, _ in vision_model.modify_tensors(tensor_loader(), source_name, None)
    ]
    if len(vision_mapped) != 439 or len(set(vision_mapped)) != 439:
        raise ValueError(
            f"Expected 439 unique vision GGUF tensors, got {len(vision_mapped)} total and "
            f"{len(set(vision_mapped))} unique"
        )

    print(f"architecture: {architecture}")
    print(f"source tensors: {len(model.model_tensors)}")
    print(f"main GGUF tensors: {len(mapped)}")
    print(f"vision GGUF tensors: {len(vision_mapped)}")
    print(f"main-deferred/skipped tensors: {len(skipped)}")
    print("mapping validation: OK")


if __name__ == "__main__":
    main()
