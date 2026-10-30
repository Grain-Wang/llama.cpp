#!/usr/bin/env python3

import argparse
import hashlib
import json

from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


EPISODE_IDS = (1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
DATASET_REVISION = "0eabc778f959c54b8c5aa3626cc1128d2d2e54d4"
CAMERAS = {
    "base_0_rgb.jpg": "observation.images.exterior_1_left",
    "left_wrist_0_rgb.jpg": "observation.images.wrist_left",
    "right_wrist_0_rgb.jpg": "observation.images.exterior_2_left",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quantile_normalize(values: np.ndarray, stats: dict[str, list[float]]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    normalized = 2.0 * (values - q01) / (q99 - q01) - 1.0
    return np.clip(normalized, -1.0, 1.0)


def load_episode_specs(metadata_root: Path) -> list[dict]:
    path = metadata_root / "meta/episodes/chunk-000/file-000.parquet"
    columns = [
        "episode_index",
        "tasks",
        "length",
        "dataset_from_index",
        "videos/observation.images.wrist_left/from_timestamp",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    selected = []
    for row in rows:
        episode_id = row["episode_index"]
        if episode_id not in EPISODE_IDS:
            continue
        frame_index = row["length"] // 2
        selected.append(
            {
                "episode_id": episode_id,
                "task": row["tasks"][0],
                "length": row["length"],
                "frame_index": frame_index,
                "dataset_index": row["dataset_from_index"] + frame_index,
                "video_timestamp": row["videos/observation.images.wrist_left/from_timestamp"] + frame_index / 15.0,
            }
        )
    if tuple(item["episode_id"] for item in selected) != EPISODE_IDS:
        raise RuntimeError("DROID episode metadata does not contain the expected cases")
    return selected


def extract_camera(video_path: Path, output_root: Path, specs: list[dict], output_name: str) -> None:
    targets = iter(specs)
    target = next(targets, None)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if target is None:
                break
            timestamp = float(frame.pts * stream.time_base)
            if timestamp + 1e-6 < target["video_timestamp"]:
                continue
            case_dir = output_root / f"case_{target['episode_id']:04d}"
            frame.to_image().save(case_dir / output_name, quality=95)
            target = next(targets, None)
    if target is not None:
        raise RuntimeError(f"video ended before episode {target['episode_id']} was extracted")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    specs = load_episode_specs(args.metadata_root)
    stats = json.loads((args.metadata_root / "meta/stats.json").read_text())
    data_path = args.shard_root / "data/chunk-000/file-000.parquet"
    data = pq.read_table(
        data_path,
        columns=["observation.state", "action", "timestamp", "frame_index", "episode_index"],
    ).to_pydict()

    for spec in specs:
        case_dir = args.output / f"case_{spec['episode_id']:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        row = spec["dataset_index"]
        state_raw = np.asarray(data["observation.state"][row], dtype=np.float32)
        state_normalized = quantile_normalize(state_raw, stats["observation.state"])
        state_padded = np.pad(state_normalized, (0, 32 - state_normalized.size))

        case = {
            "source_episode_index": spec["episode_id"],
            "source_frame_index": int(data["frame_index"][row]),
            "source_timestamp": float(data["timestamp"][row]),
            "task": spec["task"],
            "state_raw_8": state_raw.tolist(),
            "state_normalized_padded_32": state_padded.tolist(),
            "camera_mapping": CAMERAS,
        }
        (case_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n")

        action_rows = []
        episode_end = row - spec["frame_index"] + spec["length"]
        for action_row in range(row, min(row + 50, episode_end)):
            action_rows.append(data["action"][action_row])
        while len(action_rows) < 50:
            action_rows.append(action_rows[-1])
        action_raw = np.asarray(action_rows, dtype=np.float32)
        action_normalized = quantile_normalize(action_raw, stats["action"])
        action_padded = np.pad(action_normalized, ((0, 0), (0, 32 - action_normalized.shape[1])))
        action_reference = {
            "note": "DROID demonstration actions are labels for data inspection, not expected outputs from the base PI0.5 checkpoint.",
            "action_raw_50x8": action_raw.tolist(),
            "action_normalized_padded_50x32": action_padded.tolist(),
        }
        (case_dir / "action_reference.json").write_text(json.dumps(action_reference, indent=2) + "\n")

    for output_name, source_key in CAMERAS.items():
        video_path = args.shard_root / f"videos/{source_key}/chunk-000/file-000.mp4"
        extract_camera(video_path, args.output, specs, output_name)

    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    manifest = {
        "dataset": "lerobot/droid_1.0.1",
        "revision": DATASET_REVISION,
        "episode_ids": list(EPISODE_IDS),
        "selection": "middle frame from 15 episodes with non-empty language instructions",
        "files": {str(path.relative_to(args.output)): sha256(path) for path in files},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
