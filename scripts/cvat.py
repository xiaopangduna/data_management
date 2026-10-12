"""Publish existing detections to CVAT so people can correct them.

Selects samples by sample tags, uploads the current ``--label-field`` boxes for
``--classes``, and records the run under ``--anno-key``. Other classes stay in
FiftyOne and are not sent. After a successful upload, each selected sample is
tagged ``cvat_<anno-key>``.

Images are split across CVAT tasks so each upload stays under the server body
limit. Those tasks share one project and one ``--anno-key``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import fiftyone as fo

logger = logging.getLogger(__name__)

DEFAULT_LABEL_FIELD = "ground_truth"
ANNO_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# CVAT nginx rejects bodies over 1 GiB. Leave room for multipart framing.
UPLOAD_BUDGET_BYTES = 800 * 1024 * 1024
# CVAT keeps one temp file open per image in a single upload. Its process limit is 1024.
MAX_FILES_PER_TASK = 512


def nonempty(value: str) -> str:
    """Reject blank CLI strings."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return stripped


def name_list(value: str) -> list[str]:
    """Parse a comma-separated list, preserving first-seen order."""
    names = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not names:
        raise argparse.ArgumentTypeError("provide at least one name")
    return names


def anno_key(value: str) -> str:
    """Accept an annotation key that is also safe to use in a sample tag."""
    key = value.strip()
    if not ANNO_KEY_PATTERN.fullmatch(key):
        raise argparse.ArgumentTypeError("use letters, digits, underscore, dot, or hyphen")
    return key


def select_attribute(value: str) -> tuple[str, dict]:
    """Parse ``name=option1,option2`` into one CVAT select attribute."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("use name=option1,option2")
    name, raw = value.split("=", 1)
    name = name.strip()
    options = list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    if not name or not options:
        raise argparse.ArgumentTypeError("use name=option1,option2")
    if name == "label_id":
        raise argparse.ArgumentTypeError("label_id is reserved")
    return name, {"type": "select", "values": options, "mutable": True}


def attribute_schema(items: list[tuple[str, dict]]) -> dict[str, dict]:
    """Merge parsed attributes. A repeated name is rejected."""
    schema: dict[str, dict] = {}
    for name, spec in items:
        if name in schema:
            raise ValueError(f"Duplicate attribute: {name}")
        schema[name] = spec
    return schema


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for a CVAT publish run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument(
        "--sample-tags",
        required=True,
        type=name_list,
        help="Require ALL of these sample tags.",
    )
    parser.add_argument(
        "--anno-key",
        required=True,
        type=anno_key,
        help="New FiftyOne annotation run name. Also used as the CVAT task name.",
    )
    parser.add_argument(
        "--classes",
        required=True,
        type=name_list,
        help="Detection labels sent to CVAT and available for editing.",
    )
    parser.add_argument(
        "--project-name",
        default=None,
        type=nonempty,
        help="CVAT project that receives this task. Default: the dataset name.",
    )
    parser.add_argument(
        "--label-field",
        default=DEFAULT_LABEL_FIELD,
        type=nonempty,
        help=f"Detections field to publish (default: {DEFAULT_LABEL_FIELD}).",
    )
    parser.add_argument(
        "--attributes",
        action="append",
        default=[],
        type=select_attribute,
        help="Box field sent to CVAT as a dropdown: name=option1,option2. Repeat to send more fields.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected samples and exit before contacting CVAT.",
    )
    return parser.parse_args(argv)


def pending_tag(anno_key_value: str) -> str:
    """Return the sample tag that marks a published annotation run."""
    return f"cvat_{anno_key_value}"


def resolve_project_name(dataset_name: str, project_name: str | None) -> str:
    """Use the dataset name when no CVAT project name was given."""
    return project_name or dataset_name


def ensure_project_classes(existing: set[str] | None, requested: list[str]) -> None:
    """Require an existing CVAT project to already use exactly ``requested``.

    FiftyOne adopts the existing project's labels and ignores ``--classes`` on
    later uploads. Matching here keeps each publish limited to the requested labels.
    """
    if existing is None or existing == set(requested):
        return
    current = ",".join(sorted(existing))
    wanted = ",".join(requested)
    raise ValueError(
        f"CVAT project labels ({current}) differ from --classes ({wanted}). "
        "Use the same class list, or choose another --project-name."
    )


def cvat_credentials() -> tuple[str, str, str]:
    """Read the CVAT URL and login from the environment."""
    values = {
        "FIFTYONE_CVAT_URL": os.environ.get("FIFTYONE_CVAT_URL", "").strip(),
        "FIFTYONE_CVAT_USERNAME": os.environ.get("FIFTYONE_CVAT_USERNAME", "").strip(),
        "FIFTYONE_CVAT_PASSWORD": os.environ.get("FIFTYONE_CVAT_PASSWORD", ""),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError("Set " + ", ".join(missing))
    return values["FIFTYONE_CVAT_URL"], values["FIFTYONE_CVAT_USERNAME"], values["FIFTYONE_CVAT_PASSWORD"]


def project_label_attributes(
    url: str, username: str, password: str, project_name: str
) -> dict[str, set[str]] | None:
    """Return each label's attribute names, or None when the project does not exist."""
    from fiftyone.utils.cvat import CVATAnnotationAPI

    api = CVATAnnotationAPI("cvat", url, username=username, password=password)
    project_id = api.get_project_id(project_name)
    if project_id is None:
        return None
    found: dict[str, set[str]] = {}
    for label in api._get_project_labels(project_id):
        names = set()
        for attr in label.get("attributes") or []:
            attr_name = str(attr.get("name", ""))
            if attr_name and attr_name != "label_id":
                names.add(attr_name)
        found[str(label["name"])] = names
    return found


def project_label_names(url: str, username: str, password: str, project_name: str) -> set[str] | None:
    """Return labels of an existing CVAT project, or None when it does not exist."""
    found = project_label_attributes(url, username, password, project_name)
    if found is None:
        return None
    return set(found)


def ensure_project_attributes(
    existing: dict[str, set[str]] | None,
    classes: list[str],
    attributes: dict[str, dict],
) -> None:
    """Require an existing project to already define each requested box field.

    FiftyOne keeps the project's label spec and does not add attributes on a
    later upload. A new ``--project-name`` creates the dropdowns with the task.
    """
    if existing is None or not attributes:
        return
    missing = []
    for class_name in classes:
        absent = [name for name in attributes if name not in existing.get(class_name, set())]
        if absent:
            missing.append(f"{class_name}: {','.join(absent)}")
    if not missing:
        return
    detail = "; ".join(missing)
    raise ValueError(
        f"CVAT project is missing attributes ({detail}). "
        "Choose another --project-name so the new project is created with these attributes."
    )


def ensure_detections_field(dataset: fo.Dataset, label_field: str) -> None:
    """Require ``label_field`` to already store detections."""
    if not dataset.has_field(label_field):
        raise ValueError(f"Label field does not exist: {label_field}")
    field = dataset.get_field(label_field)
    if not isinstance(field, fo.EmbeddedDocumentField) or field.document_type is not fo.Detections:
        raise ValueError(f"Label field must contain Detections: {label_field}")


def file_sizes(paths: list[str]) -> list[int]:
    """Return on-disk sizes in ``paths`` order."""
    sizes = []
    for path in paths:
        try:
            size = Path(path).stat().st_size
        except OSError as error:
            raise ValueError(f"Cannot read image size for {path}: {error}") from error
        sizes.append(size)
    return sizes


def chunks_fit(sizes: list[int], count: int, budget: int) -> bool:
    """Return True when every non-overlapping group of ``count`` files stays within ``budget``."""
    for start in range(0, len(sizes), count):
        if sum(sizes[start : start + count]) > budget:
            return False
    return True


def task_size_for_budget(
    sizes: list[int],
    budget: int = UPLOAD_BUDGET_BYTES,
    max_files: int = MAX_FILES_PER_TASK,
) -> int:
    """Return how many images FiftyOne can put in one CVAT task.

    FiftyOne splits a run into equal counts, in the same order as ``sizes``.
    Each task must stay within ``budget`` bytes and ``max_files`` images.
    """
    if budget < 1:
        raise ValueError("Upload budget must be positive")
    if max_files < 1:
        raise ValueError("Max files per task must be positive")
    if not sizes:
        raise ValueError("No files to upload")
    largest = max(sizes)
    if largest > budget:
        raise ValueError(
            f"An image is {largest} bytes, above the {budget} byte CVAT upload budget."
        )
    low = 1
    high = min(len(sizes), max_files)
    best = 1
    while low <= high:
        mid = (low + high) // 2
        if chunks_fit(sizes, mid, budget):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def install_quiet_cvat_errors() -> None:
    """Replace CVAT error logging so a failed upload does not print image bytes."""
    from fiftyone.utils.cvat import CVATAnnotationAPI

    def _validate(self, response, kwargs):
        del self, kwargs
        try:
            response.raise_for_status()
        except Exception as error:
            status = getattr(response, "status_code", "?")
            url = getattr(response, "url", "")
            detail = ""
            try:
                detail = (response.text or "")[:300]
            except Exception:
                detail = ""
            raise RuntimeError(f"CVAT request failed: {status} {url} {detail}") from error

    CVATAnnotationAPI._validate = _validate


def task_count(sample_count: int, task_size: int) -> int:
    """Return how many CVAT tasks a fixed ``task_size`` produces."""
    return (sample_count + task_size - 1) // task_size


def select_view(dataset: fo.Dataset, sample_tags: list[str]) -> fo.DatasetView:
    """Return samples that have every requested tag."""
    return dataset.match_tags(sample_tags, bool=True, all=True)


def append_tag(dataset: fo.Dataset, view: fo.DatasetView, tag: str) -> int:
    """Add ``tag`` to samples that do not already have it. Returns samples updated."""
    updated = 0
    for sample in view.iter_samples(autosave=True):
        tags = [str(item) for item in (sample.tags or [])]
        if tag in tags:
            continue
        sample.tags = [*tags, tag]
        updated += 1
    dataset.save()
    return updated


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def run(args: argparse.Namespace) -> int:
    """Select samples and upload them to CVAT."""
    if not fo.dataset_exists(args.dataset):
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    dataset = fo.load_dataset(args.dataset)
    ensure_detections_field(dataset, args.label_field)
    if args.anno_key in dataset.list_annotation_runs():
        raise ValueError(
            f"Annotation run '{args.anno_key}' already exists. Choose another --anno-key."
        )
    view = select_view(dataset, args.sample_tags)
    project_name = resolve_project_name(args.dataset, args.project_name)
    tag = pending_tag(args.anno_key)
    attributes = attribute_schema(args.attributes)
    selected = len(view)
    if selected == 0:
        raise ValueError("No samples matched --sample-tags.")
    sizes = file_sizes([str(path) for path in view.values("filepath")])
    task_size = task_size_for_budget(sizes)
    tasks = task_count(selected, task_size)
    print_report(
        {
            "mode": "dry-run" if args.dry_run else "publish",
            "dataset": args.dataset,
            "sample_tags": ",".join(args.sample_tags),
            "classes": ",".join(args.classes),
            "attributes": ";".join(
                f"{name}={','.join(spec['values'])}" for name, spec in attributes.items()
            ),
            "anno_key": args.anno_key,
            "project_name": project_name,
            "task_name": args.anno_key,
            "label_field": args.label_field,
            "samples": selected,
            "upload_bytes": sum(sizes),
            "upload_budget": UPLOAD_BUDGET_BYTES,
            "task_size": task_size,
            "tasks": tasks,
            "pending_tag": tag,
        }
    )
    if args.dry_run:
        return 0

    install_quiet_cvat_errors()
    url, username, password = cvat_credentials()
    existing_attributes = project_label_attributes(url, username, password, project_name)
    existing_classes = None if existing_attributes is None else set(existing_attributes)
    ensure_project_classes(existing_classes, args.classes)
    ensure_project_attributes(existing_attributes, args.classes, attributes)
    view.annotate(
        args.anno_key,
        backend="cvat",
        label_field=args.label_field,
        label_type="detections",
        classes=args.classes,
        attributes=attributes or False,
        url=url,
        username=username,
        password=password,
        project_name=project_name,
        task_name=args.anno_key,
        task_size=task_size,
        launch_editor=False,
    )
    tagged = append_tag(dataset, view, tag)
    print_report({"tagged": tagged, "cvat_url": url})
    return 0


def main(argv: list[str] | None = None) -> int:
    """Publish a tagged slice of a dataset to CVAT."""
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
