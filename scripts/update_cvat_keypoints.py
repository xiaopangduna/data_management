"""Write 42-point CVAT face edits back into FiftyOne.

Replaces one ``kpt`` label in the keypoints field and its box in the detection
field recorded at upload. The CVAT rectangle ``<label>_box`` is stored under
the original label. Other labels in both fields stay. A point CVAT marks
outside comes back as NaN and keeps its index. Deleting the box or the whole
skeleton removes that pair. Samples whose box or points moved gain
``--sample-tags`` and ``changed``. The ``cvat_<anno-key>`` tag from the upload
is left in place. ``--dry-run`` downloads the task and prints how many samples
changed, without writing the dataset.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import fiftyone as fo

logger = logging.getLogger(__name__)
CHANGED_TAG = "changed"
PIXEL_TOLERANCE = 5.0


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


def parse_args(argv: list[str] | None = None):
    """Parse CLI arguments for writing a 42-point CVAT run back into FiftyOne."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=boxes.nonempty)
    parser.add_argument("--anno-key", required=True, type=boxes.nonempty)
    parser.add_argument("--sample-tags", required=True, type=boxes.name_list)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download CVAT and print changed and unchanged counts without writing.",
    )
    return parser.parse_args(argv)


def run_record(dataset: fo.Dataset, anno_key: str) -> dict:
    """Return the upload record for ``anno_key``."""
    record = (dataset.info or {}).get(shapes.RUNS_KEY, {}).get(anno_key)
    if not isinstance(record, dict):
        raise ValueError(f"Keypoint upload does not exist: {anno_key}")
    if record.get("keypoints_field") != shapes.KEYPOINTS_FIELD:
        raise ValueError(f"Annotation run '{anno_key}' is not a 42-point face upload")
    shapes.require_kpt_label(str(record.get("label") or ""))
    if not record.get("label_field"):
        raise ValueError(f"Annotation run '{anno_key}' is missing label_field")
    return record


def connect(api: object | None):
    """Return a CVAT client, using the environment when one was not injected."""
    if api is not None:
        return api
    from fiftyone.utils.cvat import CVATAnnotationAPI

    boxes.install_quiet_cvat_errors()
    url, username, password = boxes.cvat_credentials()
    return CVATAnnotationAPI("cvat", url, username=username, password=password)


def task_face_ids(
    api: object,
    task_id: int,
    names: list[str],
    project_id: int,
    skeleton_name: str,
    box_name: str,
) -> tuple[int, list[int], int]:
    """Return skeleton ids and the rectangle id, falling back to the project definition."""
    found = shapes.face_project_labels(api._get_task_labels(task_id), skeleton_name, box_name)
    if found is None:
        found = shapes.face_project_labels(api._get_project_labels(project_id), skeleton_name, box_name)
    if found is None:
        raise ValueError("CVAT project is missing the keypoint skeleton or its box")
    skeleton, rectangle = found
    sublabels = list(skeleton.get("sublabels") or [])
    if [str(item.get("name", "")) for item in sublabels] != list(names):
        raise ValueError("CVAT skeleton no longer matches the uploaded 42-point order")
    return int(skeleton["id"]), shapes.sublabel_ids(skeleton), int(rectangle["id"])


def downloaded_faces(
    api: object,
    tasks: list[dict],
    names: list[str],
    project_id: int,
    sizes: dict[str, tuple[int, int]],
    skeleton_name: str,
    box_name: str,
) -> dict[str, list[tuple[list, list, list]]]:
    """Return ``sample id -> [(box, points, visibility), ...]`` from CVAT."""
    found: dict[str, list[tuple[list, list, list]]] = {}
    for task in tasks:
        skeleton_id, node_order, box_id = task_face_ids(
            api, int(task["task_id"]), names, project_id, skeleton_name, box_name
        )
        payload = api.get(api.task_annotation_url(task["task_id"])).json()
        task_shapes = list(payload.get("shapes") or [])
        for frame, sample_id in enumerate(task["sample_ids"]):
            width, height = sizes[sample_id]
            faces = shapes.skeletons_from_shapes(
                task_shapes, frame, skeleton_id, node_order, width, height
            )
            boxes_found = shapes.boxes_from_shapes(task_shapes, frame, box_id, width, height)
            found[sample_id] = shapes.pair_downloaded(faces, boxes_found)
    return found


def previous_faces(detections: object, keypoints: object, label_name: str) -> list[dict]:
    """Return the stored pairs that a download will replace."""
    snapshots = []
    for box, keypoint in shapes.face_pairs(detections, keypoints, label_name):
        snapshots.append({
            "box": [float(value) for value in box.bounding_box],
            "points": [tuple(point) for point in (keypoint.points or [])],
            "instance_id": shapes.instance_id(box) or shapes.instance_id(keypoint),
            "tags": [str(tag) for tag in (box.tags or [])],
        })
    return snapshots


def assign_previous(
    previous: list[dict],
    downloaded: list[tuple[list, list, list]],
    width: int,
    height: int,
) -> list[dict | None]:
    """Match each downloaded face to a stored face so its instance id can be kept."""
    used: set[int] = set()
    assigned: list[dict | None] = [None] * len(downloaded)
    for index, (box, _points, _visibility) in enumerate(downloaded):
        for previous_index, face in enumerate(previous):
            if previous_index in used:
                continue
            if not shapes.box_moved(face["box"], box, width, height, PIXEL_TOLERANCE):
                used.add(previous_index)
                assigned[index] = face
                break
    for index, (_box, points, _visibility) in enumerate(downloaded):
        if assigned[index] is not None:
            continue
        for previous_index, face in enumerate(previous):
            if previous_index in used:
                continue
            if not shapes.points_moved(face["points"], points, width, height, PIXEL_TOLERANCE):
                used.add(previous_index)
                assigned[index] = face
                break
    remaining_previous = [index for index in range(len(previous)) if index not in used]
    remaining_downloaded = [index for index, face in enumerate(assigned) if face is None]
    if len(remaining_previous) == 1 and len(remaining_downloaded) == 1:
        assigned[remaining_downloaded[0]] = previous[remaining_previous[0]]
    return assigned


def pairs_changed(
    previous: list[dict],
    downloaded: list[tuple[list, list, list]],
    assigned: list[dict | None],
    width: int,
    height: int,
) -> bool:
    """Return whether the face boxes or points changed beyond pixel rounding."""
    if len(previous) != len(downloaded) or any(face is None for face in assigned):
        return True
    for face, (box, points, _visibility) in zip(assigned, downloaded, strict=True):
        if face is None:
            return True
        if shapes.box_moved(face["box"], box, width, height, PIXEL_TOLERANCE):
            return True
        if shapes.points_moved(face["points"], points, width, height, PIXEL_TOLERANCE):
            return True
    return False


def build_pair(label_name: str, box: list, points: list, visibility: list, previous: dict | None):
    """Return a detection and keypoint that share one instance."""
    instance_id = None if previous is None else previous.get("instance_id")
    tags = [] if previous is None else list(previous.get("tags") or [])
    instance = fo.Instance(id=instance_id) if instance_id else fo.Instance()
    detection = fo.Detection(
        label=label_name,
        bounding_box=[float(value) for value in box],
        instance=instance,
        tags=tags,
    )
    keypoint = fo.Keypoint(
        label=label_name,
        points=[tuple(point) for point in points],
        instance=instance,
    )
    keypoint["visibility"] = list(visibility)
    return detection, keypoint


def sample_sizes(dataset: fo.Dataset, sample_ids: list[str]) -> dict[str, tuple[int, int]]:
    """Return each sample's image width and height."""
    view = dataset.select(sample_ids)
    sizes = {}
    for sample_id, metadata in zip(view.values("id"), view.values("metadata"), strict=True):
        size = None
        if metadata is not None and metadata.width and metadata.height:
            size = (int(metadata.width), int(metadata.height))
        if size is None:
            raise ValueError(f"Sample {sample_id} is missing metadata width and height")
        sizes[str(sample_id)] = size
    return sizes


def count_changes(
    dataset: fo.Dataset,
    record: dict,
    downloaded: dict[str, list[tuple[list, list, list]]],
    sizes: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    """Compare downloaded faces with stored faces without writing."""
    field_name = record["keypoints_field"]
    label_field = record["label_field"]
    label_name = record["label"]
    sample_ids = [sample_id for task in record["tasks"] for sample_id in task["sample_ids"]]
    has_points = dataset.has_field(field_name)
    has_boxes = dataset.has_field(label_field)
    changed = 0
    unchanged = 0
    for sample in dataset.select(sample_ids):
        sample_id = str(sample.id)
        existing_points = sample[field_name] if has_points else None
        existing_boxes = sample[label_field] if has_boxes else None
        width, height = sizes[sample_id]
        arrived = downloaded.get(sample_id, [])
        previous = previous_faces(existing_boxes, existing_points, label_name)
        assigned = assign_previous(previous, arrived, width, height)
        if pairs_changed(previous, arrived, assigned, width, height):
            changed += 1
        else:
            unchanged += 1
    return changed, unchanged


def apply_pairs(
    dataset: fo.Dataset,
    record: dict,
    downloaded: dict[str, list[tuple[list, list, list]]],
    sizes: dict[str, tuple[int, int]],
    extra_tags: list[str],
) -> tuple[int, int]:
    """Replace face boxes and keypoints, and tag samples that changed."""
    field_name = record["keypoints_field"]
    label_field = record["label_field"]
    label_name = record["label"]
    sample_ids = [sample_id for task in record["tasks"] for sample_id in task["sample_ids"]]
    changed = 0
    unchanged = 0
    if f"{field_name}.keypoints.visibility" not in dataset.get_field_schema(flat=True):
        dataset.add_sample_field(f"{field_name}.keypoints.visibility", fo.ListField, subfield=fo.IntField)
    for sample in dataset.select(sample_ids).iter_samples(autosave=True):
        sample_id = str(sample.id)
        existing_points = sample[field_name] if dataset.has_field(field_name) else None
        existing_boxes = sample[label_field] if dataset.has_field(label_field) else None
        width, height = sizes[sample_id]
        arrived = downloaded.get(sample_id, [])
        previous = previous_faces(existing_boxes, existing_points, label_name)
        assigned = assign_previous(previous, arrived, width, height)
        replacements = [
            build_pair(label_name, box, points, visibility, face)
            for (box, points, visibility), face in zip(arrived, assigned, strict=True)
        ]
        kept_boxes = [
            detection for detection in shapes.label_items(existing_boxes, "detections")
            if getattr(detection, "label", None) != label_name
        ]
        kept_points = [
            keypoint for keypoint in shapes.label_items(existing_points, "keypoints")
            if getattr(keypoint, "label", None) != label_name
        ]
        sample[label_field] = fo.Detections(
            detections=[*kept_boxes, *[detection for detection, _keypoint in replacements]]
        )
        sample[field_name] = fo.Keypoints(
            keypoints=[*kept_points, *[keypoint for _detection, keypoint in replacements]]
        )
        if pairs_changed(previous, arrived, assigned, width, height):
            tags = [str(tag) for tag in (sample.tags or [])]
            for tag in [*extra_tags, CHANGED_TAG]:
                if tag not in tags:
                    tags.append(tag)
            sample.tags = tags
            changed += 1
        else:
            unchanged += 1
    dataset.save()
    return changed, unchanged


def run(args, api: object | None = None) -> int:
    """Download one 42-point skeleton run and merge it into FiftyOne."""
    if not fo.dataset_exists(args.dataset):
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    dataset = fo.load_dataset(args.dataset)
    record = run_record(dataset, args.anno_key)
    sample_ids = [sample_id for task in record["tasks"] for sample_id in task["sample_ids"]]
    if not sample_ids:
        raise ValueError(f"Annotation run '{args.anno_key}' has no samples.")
    boxes.print_report({
        "mode": "dry-run" if args.dry_run else "update",
        "dataset": args.dataset,
        "anno_key": args.anno_key,
        "keypoints_field": record["keypoints_field"],
        "label_field": record["label_field"],
        "label": record["label"],
        "box_label": record.get("box_label") or shapes.box_label_name(record["label"]),
        "points": shapes.POINT_COUNT,
        "project_name": record["project_name"],
        "tasks": len(record["tasks"]),
        "samples": len(sample_ids),
        "sample_tags": ",".join(args.sample_tags),
    })
    sizes = sample_sizes(dataset, sample_ids)
    api = connect(api)
    box_name = record.get("box_label") or shapes.box_label_name(record["label"])
    downloaded = downloaded_faces(
        api,
        record["tasks"],
        record["labels"],
        int(record["project_id"]),
        sizes,
        record["label"],
        box_name,
    )
    if args.dry_run:
        changed, unchanged = count_changes(dataset, record, downloaded, sizes)
    else:
        changed, unchanged = apply_pairs(dataset, record, downloaded, sizes, args.sample_tags)
    boxes.print_report({"changed": changed, "unchanged": unchanged})
    return 0


def main(argv: list[str] | None = None) -> int:
    """Write CVAT skeleton edits back into an existing dataset."""
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
