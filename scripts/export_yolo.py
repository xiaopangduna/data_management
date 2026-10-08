"""Export the union of sample tags as a single YOLO detection dataset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


def nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def names(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",")]
    if not all(items):
        raise argparse.ArgumentTypeError("provide comma-separated nonempty names")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("duplicate names are not allowed")
    return items


ATTRIBUTE_CLASS = re.compile(
    r"^(?P<label>[^-]+)-(?P<attr>[A-Za-z][A-Za-z0-9]*)_(?P<code>\d+)$"
)


def split_name(value: str) -> str:
    value = nonempty(value)
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("must be a single folder name")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--label-field", default="ground_truth", type=nonempty)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="train", type=split_name,
                        help="Folder name under images/ and labels/. Default: train.")
    parser.add_argument("--sample-tags", required=True, type=names,
                        help="Comma-separated sample tags; match ANY tag (union).")
    parser.add_argument("--exclude-tags", type=names, default=["dup_repeat_drop", "dup_near_drop"],
                        help="Exclude samples with ANY of these tags. Default: dup_repeat_drop,dup_near_drop.")
    parser.add_argument(
        "--classes", type=names,
        help=(
            "Classes in class ID order. Plain names match detection.label. "
            "Names like head-age_0 use that label plus label tags age_<n>: "
            "the highest n wins, and no such tag exports as age_0 when that class is listed. "
            "Boxes that resolve to an unlisted class are omitted. "
            "One attribute per export. Omit to export every class, sorted."
        ),
    )
    parser.add_argument("--export-media", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print the plan without writing files.")
    return parser.parse_args(argv)


@dataclass(frozen=True)
class AttributeExport:
    """YOLO classes expanded from detection labels plus one tag attribute."""

    attribute: str
    default_value: str
    labels: frozenset[str]
    by_label_code: dict[tuple[str, int], str]


def parse_attribute_classes(classes: list[str] | None) -> AttributeExport | None:
    """Return attribute rules, or None when ``--classes`` are plain labels."""
    if not classes:
        return None
    matches = [ATTRIBUTE_CLASS.fullmatch(name) for name in classes]
    if not any(matches):
        return None
    if not all(matches):
        raise ValueError(
            "--classes cannot mix plain names with attribute names such as head-age_0"
        )
    parsed = [match for match in matches if match is not None]
    attributes = {match.group("attr") for match in parsed}
    if len(attributes) != 1:
        joined = ", ".join(sorted(attributes))
        raise ValueError(f"one --classes export accepts a single attribute, got {joined}")
    by_label_code: dict[tuple[str, int], str] = {}
    for match in parsed:
        key = (match.group("label"), int(match.group("code")))
        if key in by_label_code:
            raise ValueError(
                f"duplicate attribute code for {key[0]}: {by_label_code[key]} and {match.group(0)}"
            )
        by_label_code[key] = match.group(0)
    attribute = next(iter(attributes))
    return AttributeExport(
        attribute=attribute,
        default_value=f"{attribute}_0",
        labels=frozenset(label for label, _code in by_label_code),
        by_label_code=by_label_code,
    )


def attribute_code(tags, attribute: str) -> tuple[int, bool]:
    """Return the highest ``attribute_<n>`` tag, or ``(0, True)`` when none exist."""
    prefix = f"{attribute}_"
    codes: list[int] = []
    for tag in tags or []:
        if not isinstance(tag, str) or not tag.startswith(prefix):
            continue
        suffix = tag[len(prefix):]
        if suffix.isdigit():
            codes.append(int(suffix))
    if not codes:
        return 0, True
    return max(codes), False


def apply_attribute_labels(label, spec: AttributeExport):
    """Copy in-scope boxes to YOLO class names. The input label is left unchanged."""
    import fiftyone as fo

    if label is None:
        return None, 0, 0, []
    detections = getattr(label, "detections", None)
    if not detections:
        return label, 0, 0, []
    kept = []
    defaulted = 0
    omitted_default = 0
    unmapped: list[tuple[int, str]] = []
    for index, detection in enumerate(detections):
        if detection.label not in spec.labels:
            continue
        code, used_default = attribute_code(getattr(detection, "tags", None), spec.attribute)
        yolo_label = spec.by_label_code.get((detection.label, code))
        if yolo_label is None:
            if used_default:
                omitted_default += 1
            else:
                unmapped.append((index, f"{detection.label}-{spec.attribute}_{code}"))
            continue
        if used_default:
            defaulted += 1
        kept.append(fo.Detection(label=yolo_label, bounding_box=list(detection.bounding_box)))
    return fo.Detections(detections=kept), defaulted, omitted_default, unmapped


def inspect_samples(samples, label_field: str, attribute_export: AttributeExport | None) -> dict:
    """Count exportable boxes. Attribute rules read label tags and do not modify samples."""
    detected_classes: set[str] = set()
    boxes = negatives = 0
    defaulted_boxes = omitted_default_boxes = omitted_boxes = 0
    for sample in samples:
        if not Path(sample.filepath).is_file():
            raise ValueError(f"Missing image for sample {sample.id}: {sample.filepath}")
        labels = sample[label_field]
        detections = labels.detections if labels is not None else []
        for detection in detections:
            if not isinstance(detection.label, str) or not detection.label.strip():
                raise ValueError(f"Empty class label for sample {sample.id}")
        if attribute_export is not None:
            labels, defaulted, omitted, unmapped = apply_attribute_labels(labels, attribute_export)
            defaulted_boxes += defaulted
            omitted_default_boxes += omitted
            omitted_boxes += len(unmapped)
            detections = labels.detections if labels is not None else []
        negatives += not detections
        boxes += len(detections)
        for detection in detections:
            detected_classes.add(detection.label)
    return {
        "detected_classes": detected_classes,
        "boxes": boxes,
        "negative_images": negatives,
        "defaulted_boxes": defaulted_boxes,
        "omitted_default_boxes": omitted_default_boxes,
        "omitted_boxes": omitted_boxes,
    }


def check_output(path: Path, split: str) -> None:
    if path.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path must be a directory: {path}")
    for subdir in (path / "images" / split, path / "labels" / split):
        if subdir.is_symlink():
            raise ValueError(f"Split directory must not be a symlink: {subdir}")
        if subdir.exists() and (not subdir.is_dir() or any(subdir.iterdir())):
            raise ValueError(f"Split directory must be empty or absent: {subdir}")



def export_stem(label, classes: list[str], index: int) -> str:
    """Name a sample using its exported classes and a global sequence number."""
    present = {d.label for d in label.detections} if label is not None else set()
    ordered = [name for name in classes if name in present]
    # Bound components while preserving Unicode class names and avoiding paths.
    parts = [re.sub(r"[^\w-]+", "_", name).strip("_")[:20] or "class"
             for name in ordered[:6]]
    # Cap UTF-8 bytes as well, for filesystems with a 255-byte name limit.
    parts = [part.encode("utf-8")[:30].decode("utf-8", errors="ignore") for part in parts]
    if len(ordered) > 6:
        parts.append("more")
    prefix = "__".join(parts) if parts else "negative"
    return f"{prefix}_{index:06d}"


def make_exporter(
    output: Path,
    classes: list[str],
    export_media: str,
    split: str = "train",
    attribute_export: AttributeExport | None = None,
):
    from fiftyone.utils.yolo import YOLOv5DatasetExporter

    class NamedYOLOExporter(YOLOv5DatasetExporter):
        _sample_index = 0

        def export_sample(self, image_or_path, label, metadata=None):
            self._sample_index += 1
            if attribute_export is not None:
                label, _defaulted, _omitted, _unmapped = apply_attribute_labels(label, attribute_export)
            stem = export_stem(label, classes, self._sample_index)
            image_path = Path(self.data_path) / (stem + Path(image_or_path).suffix)
            self._media_exporter.export(image_or_path, outpath=str(image_path))
            labels_path = Path(self.labels_path) / (stem + ".txt")
            if label is None:
                labels_path.parent.mkdir(parents=True, exist_ok=True)
                labels_path.write_text("", encoding="utf-8")
            else:
                self._writer.write(
                    label, str(labels_path), self._labels_map_rev,
                    dynamic_classes=self._dynamic_classes,
                    include_confidence=self.include_confidence,
                    use_masks=self.use_masks, tolerance=self.tolerance,
                )

    return NamedYOLOExporter(
        export_dir=str(output), split=split, classes=classes,
        export_media=True if export_media == "copy" else export_media,
    )


def export_dataset(args: argparse.Namespace) -> dict:
    import fiftyone as fo

    attribute_export = parse_attribute_classes(args.classes)
    output = args.out_dir.expanduser().absolute()
    check_output(output, args.split)
    dataset = fo.load_dataset(args.dataset)
    if dataset.media_type != "image":
        raise ValueError("Only image datasets are supported")
    field = dataset.get_field_schema().get(args.label_field)
    if not isinstance(field, fo.EmbeddedDocumentField) or field.document_type != fo.Detections:
        raise ValueError(f"Label field must be fo.Detections: {args.label_field}")
    missing_tags = sorted(set(args.sample_tags) - set(dataset.distinct("tags")))
    if missing_tags:
        raise ValueError(f"Unknown sample tags: {', '.join(missing_tags)}")
    view = dataset.match_tags(args.sample_tags, all=False)
    if args.exclude_tags:
        view = view.exclude(dataset.match_tags(args.exclude_tags, all=False))
    count = len(view)
    if not count:
        raise ValueError("No samples match the requested tags")

    if args.classes is not None and attribute_export is None:
        view = view.filter_labels(
            args.label_field, fo.ViewField("label").is_in(args.classes),
            only_matches=False,
        )

    inspected = inspect_samples(view.iter_samples(), args.label_field, attribute_export)
    detected_classes = inspected["detected_classes"]
    boxes = inspected["boxes"]
    negatives = inspected["negative_images"]
    defaulted_boxes = inspected["defaulted_boxes"]
    omitted_default_boxes = inspected["omitted_default_boxes"]

    classes = args.classes if args.classes is not None else sorted(detected_classes)
    summary = {
        "dataset": args.dataset,
        "label_field": args.label_field,
        "tags": args.sample_tags,
        "exclude_tags": args.exclude_tags,
        "tag_matching": "union",
        "split": args.split,
        "output_dir": str(output),
        "export_media": args.export_media,
        "filename_scheme": "class_prefix_global_sequence",
        "classes": classes,
        "images": count,
        "boxes": boxes,
        "negative_images": negatives,
    }
    if attribute_export is not None:
        summary["attribute"] = attribute_export.attribute
        summary["default_value"] = attribute_export.default_value
        summary["defaulted_boxes"] = defaulted_boxes
        summary["omitted_default_boxes"] = omitted_default_boxes
        summary["omitted_boxes"] = inspected["omitted_boxes"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run: no files written.")
        return summary

    check_output(output, args.split)
    view.export(
        dataset_exporter=make_exporter(
            output, classes, args.export_media, args.split, attribute_export,
        ),
        label_field=args.label_field,
    )
    (output / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Export complete: {output}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        export_dataset(args)
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
