"""Convert YOLO detection TXT to X-AnyLabeling JSON without accessing FiftyOne."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def class_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",")]
    if not all(names) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("class names must be nonempty and unique, in YOLO ID order")
    return names


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images-dir", "labels-dir", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--class-names", type=class_names, required=True,
                        help="Complete comma-separated class mapping, in YOLO ID order")
    parser.add_argument("--export-media", choices=("none", "symlink", "copy"), default="none")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write issue CSV only")
    return parser.parse_args(argv)


def iter_files(root: Path, suffixes: set[str]):
    """Walk deterministically without following directory symlinks."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() in suffixes:
                yield path


def parse_shapes(label: Path, names: list[str], width: int, height: int) -> list[dict]:
    shapes = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            tokens = line.split()
            if len(tokens) != 5:
                raise ValueError("expected class_id cx cy w h (detection boxes only)")
            class_id = int(tokens[0])
            if not 0 <= class_id < len(names):
                raise ValueError("class ID outside --class-names mapping")
            cx, cy, bw, bh = map(float, tokens[1:])
            if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
                raise ValueError("coordinates must be finite")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError("invalid normalized center or size")
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            if min(x1, y1) < -1e-6 or max(x2, y2) > 1 + 1e-6:
                raise ValueError("box extends outside image")
            # Clamp only tiny floating-point/YOLO serialization boundary errors.
            points = [[max(0.0, x1) * width, max(0.0, y1) * height],
                      [min(1.0, x2) * width, min(1.0, y2) * height]]
            shapes.append({"label": names[class_id], "points": points,
                           "group_id": None, "shape_type": "rectangle", "flags": {},
                           "description": "", "difficult": False})
        except ValueError as error:
            raise ValueError(f"line {line_no}: {error}") from error
    return shapes


def atomic_output(destination: Path, writer):
    """Replace the directory entry, never follow an existing output symlink."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".xlabel-", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def convert(args) -> dict:
    images, labels, output = (getattr(args, name).expanduser().resolve()
                              for name in ("images_dir", "labels_dir", "out_dir"))
    if not images.is_dir() or not labels.is_dir():
        raise ValueError("--images-dir and --labels-dir must be existing directories")
    for source in (images, labels):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("--out-dir must be separate from image and label trees")

    image_index = defaultdict(list)
    label_index = defaultdict(list)
    for path in iter_files(images, IMAGE_SUFFIXES):
        image_index[path.relative_to(images).with_suffix("").as_posix()].append(path)
    for path in iter_files(labels, {".txt"}):
        label_index[path.relative_to(labels).with_suffix("").as_posix()].append(path)
    counts = dict(images=sum(map(len, image_index.values())),
                  label_files=sum(map(len, label_index.values())), to_write=0, written=0,
                  empty_labels=0, skipped=0)
    issues = []

    def issue(kind, relpath, detail=""):
        issues.append(dict(issue=kind, relpath=relpath, detail=str(detail)))
        counts["skipped"] += 1

    for key in sorted(image_index.keys() | label_index.keys()):
        candidates, annotations = image_index.get(key, []), label_index.get(key, [])
        if not candidates:
            issue("missing_image", key)
            continue
        if not annotations:
            issue("missing_label", key)
            continue
        if len(candidates) != 1 or len(annotations) != 1:
            issue("path_collision", key, "multiple images or TXT files share the same relative stem")
            continue
        image, label = candidates[0], annotations[0]
        relpath = image.relative_to(images)
        json_path = output / relpath.with_suffix(".json")
        media_path = output / relpath
        destinations = [json_path] + ([media_path] if args.export_media != "none" else [])
        # Reject pre-existing directory links so nested outputs cannot escape out-dir.
        if any(parent.is_symlink() for dest in destinations for parent in dest.parents
               if parent != output and output in parent.parents):
            issue("output_conflict", key, "output parent is a symlink")
            continue
        if any(dest.is_dir() for dest in destinations):
            issue("output_conflict", key, "output destination is a directory")
            continue
        if not args.overwrite and any(os.path.lexists(dest) for dest in destinations):
            issue("existing_output", key)
            continue
        try:
            with Image.open(image) as opened:
                width, height = opened.size
                opened.verify()
            shapes = parse_shapes(label, args.class_names, width, height)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            issue("invalid_input", key, error)
            continue
        document = dict(version="2.5.0", flags={}, shapes=shapes, description="",
                        checked=False, imagePath=str(image.resolve()) if args.export_media == "none" else image.name,
                        imageData=None, imageHeight=height, imageWidth=width)
        counts["to_write"] += 1
        counts["empty_labels"] += not shapes
        if args.dry_run:
            continue
        try:
            if args.export_media == "copy":
                atomic_output(media_path, lambda path: shutil.copy2(image, path))
            elif args.export_media == "symlink":
                def link(path):
                    path.unlink()
                    path.symlink_to(image.resolve())
                atomic_output(media_path, link)
            atomic_output(json_path, lambda path: path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"))
            counts["written"] += 1
        except OSError as error:
            issue("write_error", key, error)

    report = Path.cwd() / "tmp" / "convert_yolo_to_xlabel_issues.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["issue", "relpath", "detail"])
        writer.writeheader()
        writer.writerows(issues)
    return dict(mode="dry-run" if args.dry_run else "convert", **counts,
                issues=len(issues), csv_path=str(report))


def main(argv=None):
    args = parse_args(argv)
    try:
        result = convert(args)
    except (ValueError, OSError) as error:
        print(f"error={error}")
        return 1
    for key, value in result.items():
        print(f"{key}={value}")
    return int(result["issues"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
