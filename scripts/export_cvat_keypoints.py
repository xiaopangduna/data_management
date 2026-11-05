"""Upload 42-point faces and their boxes to CVAT.

Selects samples by sample tags. Each face is a CVAT skeleton plus a rectangle
in the same group. The skeleton keeps the FiftyOne label, which must start with
``kpt``. The rectangle is named ``<label>_box`` and is written back under the
original label. A sample whose boxes and keypoints of that label do not pair
stops the run. After a successful upload, each selected sample is tagged
``cvat_<anno-key>``.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import fiftyone as fo

logger = logging.getLogger(__name__)


def _load_export_cvat():
    path = Path(__file__).with_name("export_cvat.py")
    spec = importlib.util.spec_from_file_location("export_cvat_boxes", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_shapes():
    path = Path(__file__).with_name("cvat_keypoints.py")
    spec = importlib.util.spec_from_file_location("cvat_keypoints", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boxes = _load_export_cvat()
shapes = _load_shapes()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for a 42-point CVAT upload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=boxes.nonempty)
    parser.add_argument("--sample-tags", required=True, type=boxes.name_list)
    parser.add_argument("--anno-key", required=True, type=boxes.anno_key)
    parser.add_argument("--project-name", default=None, type=boxes.nonempty)
    parser.add_argument("--label-field", default=boxes.DEFAULT_LABEL_FIELD, type=boxes.nonempty)
    parser.add_argument("--keypoints-field", default=shapes.KEYPOINTS_FIELD, type=boxes.nonempty)
    parser.add_argument("--label", default=shapes.FACE_LABEL, type=boxes.nonempty)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def default_project_name(dataset_name: str) -> str:
    """Return the CVAT project used for 42-point faces."""
    return f"{dataset_name}_keypoints_42"


def require_skeleton(dataset: fo.Dataset, field_name: str) -> fo.KeypointSkeleton:
    """Return the 42-point skeleton stored on ``field_name``."""
    if not dataset.has_field(field_name):
        raise ValueError(f"Keypoints field does not exist: {field_name}")
    field = dataset.get_field(field_name)
    if not isinstance(field, fo.EmbeddedDocumentField) or field.document_type is not fo.Keypoints:
        raise ValueError(f"Field must contain Keypoints: {field_name}")
    skeleton = (dataset.skeletons or {}).get(field_name)
    if skeleton is None:
        raise ValueError(f"Field '{field_name}' has no skeleton")
    labels = list(skeleton.labels or [])
    if len(labels) != shapes.POINT_COUNT:
        raise ValueError(
            f"This upload is for {shapes.POINT_COUNT} points, found {len(labels)} on '{field_name}'"
        )
    return skeleton


def saved_runs(dataset: fo.Dataset) -> dict:
    """Return keypoint upload records stored on the dataset."""
    info = dataset.info or {}
    runs = info.get(shapes.RUNS_KEY) or {}
    return dict(runs)


def save_run(dataset: fo.Dataset, anno_key: str, record: dict) -> None:
    """Store one upload record without dropping other dataset info."""
    info = dict(dataset.info or {})
    runs = dict(info.get(shapes.RUNS_KEY) or {})
    runs[anno_key] = record
    info[shapes.RUNS_KEY] = runs
    dataset.info = info
    dataset.save()


def frame_size(metadata: object) -> tuple[int, int] | None:
    """Return ``(width, height)`` when both metadata values are present."""
    if metadata is None:
        return None
    width = getattr(metadata, "width", None)
    height = getattr(metadata, "height", None)
    if isinstance(width, bool) or isinstance(height, bool):
        return None
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return None
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def shapes_for_pairs(
    pairs: list[tuple],
    skeleton_id: int,
    sublabel_ids: list[int],
    box_id: int,
    size: tuple[int, int],
    frame: int,
) -> list[dict]:
    """Return one rectangle and one skeleton for each face. Groups start at 1."""
    width, height = size
    built = []
    for group, (box, keypoint) in enumerate(pairs, start=1):
        points = list(keypoint.points or [])
        if len(points) != shapes.POINT_COUNT:
            raise ValueError(f"Expected {shapes.POINT_COUNT} points, found {len(points)}")
        built.append(shapes.rectangle_shape(box_id, box.bounding_box, width, height, frame, group))
        built.append(shapes.skeleton_shape(
            skeleton_id, sublabel_ids, points, width, height, frame, group=group
        ))
    return built


def task_face_ids(
    task_labels: list[dict], names: list[str], skeleton_name: str, box_name: str
) -> tuple[int, list[int], int] | None:
    """Return task skeleton ids and the rectangle id, or None when the task has neither."""
    found = shapes.face_project_labels(task_labels, skeleton_name, box_name)
    if found is None:
        return None
    skeleton, rectangle = found
    sublabels = list(skeleton.get("sublabels") or [])
    if [str(item.get("name", "")) for item in sublabels] != list(names):
        raise ValueError("CVAT task skeleton order does not match the dataset")
    return int(skeleton["id"]), shapes.sublabel_ids(skeleton), int(rectangle["id"])


def task_annotations(chunk: list[dict]) -> dict:
    """Return the CVAT annotations document for one task."""
    task_shapes = [shape for item in chunk for shape in item["shapes"]]
    return {"version": 0, "tags": [], "shapes": task_shapes, "tracks": []}


def ensure_project(
    api: object, project_name: str, skeleton: fo.KeypointSkeleton, label: str
) -> tuple[int, int, list[int], int]:
    """Create or check the face project. Returns project, skeleton, sublabel, and box ids."""
    point_names = list(skeleton.labels or [])
    edges = [list(edge) for edge in skeleton.edges or []]
    box_name = shapes.box_label_name(label)
    project_id = api.get_project_id(project_name)
    if project_id is None:
        created = api.post(api.projects_url, json={
            "name": project_name,
            "labels": [
                shapes.skeleton_label(point_names, edges, name=label),
                shapes.rectangle_label(box_name),
            ],
        }).json()
        project_id = created["id"]
    found = api._get_project_labels(project_id)
    names = shapes.project_sublabel_names(found, label)
    if names != point_names:
        raise ValueError(
            f"CVAT project '{project_name}' must contain skeleton '{label}' and rectangle '{box_name}' "
            "in the dataset's 42-point order. Choose another --project-name."
        )
    skeleton_label, rectangle = shapes.face_project_labels(found, label, box_name)
    return (
        int(project_id),
        int(skeleton_label["id"]),
        shapes.sublabel_ids(skeleton_label),
        int(rectangle["id"]),
    )


def connect(api: object | None):
    """Return a CVAT client, using the environment when one was not injected."""
    if api is not None:
        return api
    from fiftyone.utils.cvat import CVATAnnotationAPI

    boxes.install_quiet_cvat_errors()
    url, username, password = boxes.cvat_credentials()
    return CVATAnnotationAPI("cvat", url, username=username, password=password)


def run(args: argparse.Namespace, api: object | None = None) -> int:
    """Select samples and upload their 42-point faces to CVAT."""
    if not fo.dataset_exists(args.dataset):
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    dataset = fo.load_dataset(args.dataset)
    if args.keypoints_field != shapes.KEYPOINTS_FIELD:
        raise ValueError(
            f"This upload sends {shapes.POINT_COUNT} points from {shapes.KEYPOINTS_FIELD}"
        )
    shapes.require_kpt_label(args.label)
    skeleton = require_skeleton(dataset, args.keypoints_field)
    boxes.ensure_detections_field(dataset, args.label_field)
    if args.anno_key in dataset.list_annotation_runs() or args.anno_key in saved_runs(dataset):
        raise ValueError(f"Annotation run '{args.anno_key}' already exists. Choose another --anno-key.")
    view = boxes.select_view(dataset, args.sample_tags)
    selected = len(view)
    if selected == 0:
        raise ValueError("No samples matched --sample-tags.")
    project_name = args.project_name or default_project_name(args.dataset)
    box_name = shapes.box_label_name(args.label)
    tag = boxes.pending_tag(args.anno_key)
    paths = [str(path) for path in view.values("filepath")]
    sizes = boxes.file_sizes(paths)
    sample_ids = [str(sample_id) for sample_id in view.values("id")]
    metadata = view.values("metadata")
    keypoints = view.values(args.keypoints_field)
    detections = view.values(args.label_field)
    prepared = []
    for sample_id, meta, points_value, boxes_value in zip(
        sample_ids, metadata, keypoints, detections, strict=True
    ):
        size = frame_size(meta)
        if size is None:
            raise ValueError(f"Sample {sample_id} is missing metadata width and height")
        try:
            pairs = shapes.face_pairs(boxes_value, points_value, args.label)
        except ValueError as error:
            raise ValueError(f"Sample {sample_id}: {error}") from error
        for _box, keypoint in pairs:
            if len(list(keypoint.points or [])) != shapes.POINT_COUNT:
                raise ValueError(
                    f"Sample {sample_id}: expected {shapes.POINT_COUNT} points, "
                    f"found {len(list(keypoint.points or []))}"
                )
        prepared.append({"sample_id": sample_id, "size": size, "pairs": pairs})
    task_size = boxes.task_size_for_budget(sizes)
    task_total = boxes.task_count(selected, task_size)
    boxes.print_report({
        "mode": "dry-run" if args.dry_run else "export",
        "dataset": args.dataset,
        "sample_tags": ",".join(args.sample_tags),
        "label": args.label,
        "box_label": box_name,
        "label_field": args.label_field,
        "keypoints_field": args.keypoints_field,
        "points": shapes.POINT_COUNT,
        "anno_key": args.anno_key,
        "project_name": project_name,
        "samples": selected,
        "pairs": sum(len(item["pairs"]) for item in prepared),
        "upload_bytes": sum(sizes),
        "task_size": task_size,
        "tasks": task_total,
        "pending_tag": tag,
    })
    if args.dry_run:
        return 0

    api = connect(api)
    project_id, project_label_id, project_sublabels, project_box_id = ensure_project(
        api, project_name, skeleton, args.label
    )
    label_names = list(skeleton.labels or [])
    tasks = []
    for start in range(0, len(prepared), task_size):
        chunk = prepared[start : start + task_size]
        task_index = len(tasks) + 1
        task_name = args.anno_key if task_total == 1 else f"{args.anno_key}_{task_index}"
        task_id, _class_map, _attr_map = api.create_task(
            task_name,
            schema={},
            segment_size=len(chunk),
            project_id=project_id,
        )
        api.upload_data(task_id, paths[start : start + task_size])
        located = task_face_ids(api._get_task_labels(task_id), label_names, args.label, box_name)
        label_id, sublabel_order, box_id = located or (project_label_id, project_sublabels, project_box_id)
        for offset, item in enumerate(chunk):
            item["shapes"] = shapes_for_pairs(
                item["pairs"], label_id, sublabel_order, box_id, item["size"], offset
            )
        api.put(api.task_annotation_url(task_id), json=task_annotations(chunk))
        tasks.append({
            "task_id": int(task_id),
            "task_name": task_name,
            "sample_ids": [item["sample_id"] for item in chunk],
        })
    save_run(dataset, args.anno_key, {
        "project_name": project_name,
        "project_id": int(project_id),
        "keypoints_field": args.keypoints_field,
        "label_field": args.label_field,
        "label": args.label,
        "box_label": box_name,
        "labels": list(skeleton.labels or []),
        "tasks": tasks,
    })
    tagged = boxes.append_tag(dataset, view, tag)
    boxes.print_report({"tagged": tagged, "project_id": project_id})
    return 0


def main(argv: list[str] | None = None) -> int:
    """Upload a tagged slice of 42-point faces to CVAT."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
