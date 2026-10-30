# PI0.5 DROID input cases

This directory contains 15 small integration cases extracted from `lerobot/droid_1.0.1` at revision `0eabc778f959c54b8c5aa3626cc1128d2d2e54d4`.

Each case contains one synchronized middle frame from three DROID cameras, a language instruction, the robot state, and a 50-step demonstration action window.

Camera slot mapping:

```text
base_0_rgb        <- observation.images.exterior_1_left
left_wrist_0_rgb  <- observation.images.wrist_left
right_wrist_0_rgb <- observation.images.exterior_2_left
```

DROID has one wrist camera and two exterior cameras. `right_wrist_0_rgb` is therefore a three-input compatibility slot, not a real right-wrist view. Tests must retain this distinction.

The DROID state and action vectors have 8 dimensions. The case metadata applies the dataset q01/q99 normalization and pads the remaining PI0.5 dimensions with zeros. The base PI0.5 checkpoint is not DROID-finetuned and does not contain DROID postprocessing statistics. These cases are suitable for input preprocessing, model loading, deterministic smoke tests, and performance tests. They are not a policy-accuracy benchmark.

The Hugging Face dataset repository labels the LeRobot conversion as Apache-2.0. The underlying DROID publication describes the original dataset as CC BY 4.0. Preserve DROID attribution when redistributing these derived images.

Sources:

- https://huggingface.co/datasets/lerobot/droid_1.0.1
- https://droid-dataset.github.io/
- https://arxiv.org/abs/2403.12945

Regenerate the cases after downloading the required DROID shard:

```bash
python scripts/pi05/extract_droid_cases.py \
  --shard-root /tmp/pi05-droid-shard \
  --metadata-root /tmp/pi05-droid-meta \
  --output tests/pi05/cases/droid
```
