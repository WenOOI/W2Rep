#!/usr/bin/env python3
"""Build portable W2Rep JSON/CSV manifests from official SSv2 metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def normalize_template(value: str) -> str:
    return value.replace("[", "").replace("]", "")


def load_split(path: Path, labels: dict[str, int], video_dir: Path) -> list[dict]:
    records = json.loads(path.read_text())
    result = []
    for record in records:
        template = normalize_template(record["template"])
        if template not in labels:
            raise KeyError(f"Unknown template in {path}: {template!r}")
        video_id = str(record["id"])
        result.append(
            {
                "id": video_id,
                "path": str(video_dir / f"{video_id}.webm"),
                "label": int(labels[template]),
                "template": template,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--relative-to",
        type=Path,
        help="Store video paths relative to this root instead of absolute paths.",
    )
    args = parser.parse_args()
    outputs = [args.output_json] + ([args.output_csv] if args.output_csv else [])
    for output in outputs:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
    raw_labels = json.loads(args.labels.read_text())
    labels = {normalize_template(key): int(value) for key, value in raw_labels.items()}
    train = load_split(args.train_json, labels, args.video_dir)
    validation = load_split(args.validation_json, labels, args.video_dir)
    if args.relative_to:
        root = args.relative_to.expanduser().resolve()
        for entry in train + validation:
            entry["path"] = str(Path(entry["path"]).expanduser().resolve().relative_to(root))
    payload = {"train": train, "val": validation, "labels": labels}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("path", "label", "split", "id", "template")
            )
            writer.writeheader()
            for split, entries in (("train", train), ("val", validation)):
                for entry in entries:
                    writer.writerow({**entry, "split": split})
    print(f"train={len(train)} val={len(validation)} classes={len(labels)}")


if __name__ == "__main__":
    main()

