"""Helpers for X-AnyLabeling attach comparison and tags."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    module_name = name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


attach = load_script("update_xlabel_labels.py")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_task_pair(
    task_dir: Path,
    stem: str,
    *,
    sample_id: str | None = None,
    image_path: str | None = None,
    image_bytes: bytes | None = None,
    suffix: str = ".jpg",
    width: int = 100,
    height: int = 100,
    shapes: list | None = None,
    subdir: str = "",
):
    folder = task_dir / subdir if subdir else task_dir
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / f"{stem}{suffix}"
    if image_bytes is None:
        Image.new("RGB", (width, height), color=(stem.encode()[0] % 200, 40, 80)).save(image)
    else:
        image.write_bytes(image_bytes)
    payload = {
        "imageWidth": width,
        "imageHeight": height,
        "shapes": shapes
        if shapes is not None
        else [
            {
                "shape_type": "rectangle",
                "label": "person",
                "points": [[10, 10], [50, 50]],
            }
        ],
    }
    if sample_id is not None:
        payload["sample_id"] = sample_id
    if image_path is not None:
        payload["imagePath"] = image_path
    else:
        payload["imagePath"] = image.name
    (folder / f"{stem}.json").write_text(json.dumps(payload))
    return image


class HashDataset:
    def __init__(self, rows: list[tuple[str, str]]):
        """rows: (sample_id, sha256)."""
        self.rows = rows
        self.name = "hash_dataset"

    def get_field_schema(self):
        return {"sha256": object()}

    def values(self, fields):
        assert fields == ["id", "sha256"]
        return [row[0] for row in self.rows], [row[1] for row in self.rows]

    def has_field(self, name):
        return name == "sha256"

    def select(self, ids):
        selected = [row for row in self.rows if row[0] in ids]
        return HashView(selected)


class HashView:
    def __init__(self, rows):
        self.rows = rows

    def values(self, fields):
        if fields == "id":
            return [row[0] for row in self.rows]
        assert fields == ["id", attach.DEFAULT_LABEL_FIELD]
        return [row[0] for row in self.rows], [None] * len(self.rows)


def test_boxes_equal_allows_two_pixels():
    names = {"person"}
    box = (("person", (0.1, 0.2, 0.3, 0.4)),)
    within = (("person", (0.12, 0.2, 0.3, 0.4)),)
    beyond = (("person", (0.13, 0.2, 0.3, 0.4)),)
    assert attach.boxes_equal(box, box, names, 100, 50)
    assert attach.boxes_equal(box, within, names, 100, 50)
    assert not attach.boxes_equal(box, beyond, names, 100, 50)
    assert attach.boxes_equal(None, (), names, 100, 50)
    assert not attach.boxes_equal(box, (), names, 100, 50)


def test_merge_tags_only_on_changed():
    assert attach.merge_tags(["val2017"], ["label_person_260909"], False, boxes_changed=True) == [
        "val2017",
        "label_person_260909",
        "changed",
    ]
    assert attach.merge_tags(
        ["val2017", "label_person_260909", "changed"],
        ["label_person_260909"],
        False,
        boxes_changed=False,
    ) == ["val2017"]


def test_cli_uses_task_dir_and_rejects_old_options():
    args = attach.parse_args(
        [
            "--dataset",
            "dataset",
            "--task-dir",
            ".",
            "--classes",
            "person",
            "--sample-tags",
            "batch",
        ]
    )
    assert args.task_dir == Path(".")
    assert args.label_field == "ground_truth"
    for option in ("--label-dir", "--images-dir"):
        with pytest.raises(SystemExit):
            attach.parse_args(
                [
                    "--dataset",
                    "dataset",
                    "--task-dir",
                    ".",
                    "--classes",
                    "person",
                    "--sample-tags",
                    "batch",
                    option,
                    ".",
                ]
            )


def test_attach_by_sample_id(tmp_path):
    task = tmp_path / "task"
    sample_id = "000000000000000000000000"
    image = write_task_pair(task, "a", sample_id=sample_id)
    digest = attach.compute_sha256_hex(image)
    dataset = HashDataset([(sample_id, digest), ("111111111111111111111111", "deadbeef")])
    plan = attach.build_attach_plan(dataset, task, ["person"])
    assert list(plan.to_write) == [sample_id]
    assert not plan.issues
    assert plan.to_write[sample_id].image_sha256 == digest
    assert plan.to_write[sample_id].image_path == "a.jpg"


def test_attach_falls_back_to_sha256(tmp_path):
    task = tmp_path / "task"
    image = write_task_pair(task, "a", sample_id=None)
    digest = attach.compute_sha256_hex(image)
    sample_id = "000000000000000000000000"
    plan = attach.build_attach_plan(HashDataset([(sample_id, digest)]), task, ["person"])
    assert list(plan.to_write) == [sample_id]
    assert not plan.issues


def test_attach_orphan_and_ambiguous_sha256(tmp_path):
    task = tmp_path / "task"
    image = write_task_pair(task, "a", sample_id=None)
    digest = attach.compute_sha256_hex(image)
    plan = attach.build_attach_plan(HashDataset([]), task, ["person"])
    assert [row["issue"] for row in plan.issues] == ["orphan_sha256"]
    assert not plan.to_write
    plan = attach.build_attach_plan(
        HashDataset(
            [
                ("000000000000000000000000", digest),
                ("111111111111111111111111", digest),
            ]
        ),
        task,
        ["person"],
    )
    assert [row["issue"] for row in plan.issues] == ["ambiguous_sha256"]
    assert "000000000000000000000000" in plan.issues[0]["detail"]
    assert "111111111111111111111111" in plan.issues[0]["detail"]
    assert not plan.to_write


def test_attach_orphan_sample_id_does_not_fall_back(tmp_path):
    task = tmp_path / "task"
    image = write_task_pair(task, "a", sample_id="000000000000000000000000")
    digest = attach.compute_sha256_hex(image)
    plan = attach.build_attach_plan(
        HashDataset([("111111111111111111111111", digest)]),
        task,
        ["person"],
    )
    assert [row["issue"] for row in plan.issues] == ["orphan_sample_id"]
    assert not plan.to_write


def test_attach_missing_image(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "a.json").write_text(
        json.dumps(
            {
                "imagePath": "missing.jpg",
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [],
            }
        )
    )
    plan = attach.build_attach_plan(
        HashDataset([("000000000000000000000000", "abc")]),
        task,
        ["person"],
    )
    assert [row["issue"] for row in plan.issues] == ["missing_image"]
    assert not plan.to_write


def test_attach_same_stem_without_image_path(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    image = task / "a.png"
    Image.new("RGB", (100, 100), color=(1, 2, 3)).save(image)
    (task / "a.json").write_text(
        json.dumps(
            {
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [
                    {
                        "shape_type": "rectangle",
                        "label": "person",
                        "points": [[10, 10], [50, 50]],
                    }
                ],
            }
        )
    )
    digest = attach.compute_sha256_hex(image)
    sample_id = "000000000000000000000000"
    plan = attach.build_attach_plan(HashDataset([(sample_id, digest)]), task, ["person"])
    assert list(plan.to_write) == [sample_id]


def test_attach_absolute_image_path_uses_basename_inside_task(tmp_path):
    task = tmp_path / "task"
    image = write_task_pair(
        task,
        "a",
        sample_id=None,
        image_path=str(tmp_path / "outside" / "a.jpg"),
    )
    digest = attach.compute_sha256_hex(image)
    sample_id = "000000000000000000000000"
    plan = attach.build_attach_plan(HashDataset([(sample_id, digest)]), task, ["person"])
    assert list(plan.to_write) == [sample_id]


def test_attach_rejects_image_path_outside_task_dir(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    outside = tmp_path / "outside" / "a.jpg"
    outside.parent.mkdir()
    Image.new("RGB", (100, 100)).save(outside)
    (task / "a.json").write_text(
        json.dumps(
            {
                "imagePath": "../outside/a.jpg",
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [],
            }
        )
    )
    plan = attach.build_attach_plan(
        HashDataset([("000000000000000000000000", "abc")]),
        task,
        ["person"],
    )
    assert [row["issue"] for row in plan.issues] == ["missing_image"]


def test_duplicate_task_image_hash_same_sample_continues(tmp_path):
    task = tmp_path / "task"
    payload = Image.new("RGB", (100, 100), color=(9, 9, 9)).tobytes()
    # Same visual via identical saved files
    image_a = write_task_pair(task, "a", sample_id="000000000000000000000000", subdir="one")
    image_b_dir = task / "two"
    image_b_dir.mkdir(parents=True)
    image_b = image_b_dir / "b.jpg"
    image_b.write_bytes(image_a.read_bytes())
    (image_b_dir / "b.json").write_text(
        json.dumps(
            {
                "sample_id": "000000000000000000000000",
                "imagePath": "b.jpg",
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [
                    {
                        "shape_type": "rectangle",
                        "label": "person",
                        "points": [[10, 10], [50, 50]],
                    }
                ],
            }
        )
    )
    digest = attach.compute_sha256_hex(image_a)
    assert digest == attach.compute_sha256_hex(image_b)
    plan = attach.build_attach_plan(
        HashDataset([("000000000000000000000000", digest)]),
        task,
        ["person"],
    )
    assert plan.matched == 1
    assert list(plan.to_write) == ["000000000000000000000000"]
    assert {row["issue"] for row in plan.issues} == {
        "duplicate_task_image_hash",
        "sample_id_collision",
    }
    del payload


def test_duplicate_task_image_hash_different_samples_skips_all(tmp_path):
    task = tmp_path / "task"
    image_a = write_task_pair(task, "a", sample_id="000000000000000000000000", subdir="one")
    image_b_dir = task / "two"
    image_b_dir.mkdir(parents=True)
    image_b = image_b_dir / "b.jpg"
    image_b.write_bytes(image_a.read_bytes())
    (image_b_dir / "b.json").write_text(
        json.dumps(
            {
                "sample_id": "111111111111111111111111",
                "imagePath": "b.jpg",
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [
                    {
                        "shape_type": "rectangle",
                        "label": "person",
                        "points": [[10, 10], [50, 50]],
                    }
                ],
            }
        )
    )
    digest = attach.compute_sha256_hex(image_a)
    plan = attach.build_attach_plan(
        HashDataset(
            [
                ("000000000000000000000000", digest),
                ("111111111111111111111111", "ffffffff"),
            ]
        ),
        task,
        ["person"],
    )
    assert not plan.to_write
    assert plan.matched == 0
    assert all(
        row["issue"] == "duplicate_task_image_hash" for row in plan.issues
    )


def test_missing_sha256_field_raises():
    class NoHashDataset:
        def get_field_schema(self):
            return {}

    with pytest.raises(ValueError, match="missing sha256"):
        attach.build_attach_plan(NoHashDataset(), Path("."), ["person"])


def test_converted_media_modes_attach_by_sha256(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    converter = load_script("convert_yolo_to_xlabel.py")
    images, labels = tmp_path / "images", tmp_path / "labels"
    (images / "train").mkdir(parents=True)
    (labels / "train").mkdir(parents=True)
    original = images / "train/a.jpg"
    Image.new("RGB", (100, 50)).save(original)
    (labels / "train/a.txt").write_text("0 .5 .5 .4 .4")
    digest = attach.compute_sha256_hex(original)
    sample_id = "000000000000000000000000"
    for mode in ("symlink", "copy"):
        output = tmp_path / mode
        assert (
            converter.main(
                [
                    "--images-dir",
                    str(images),
                    "--label-dir",
                    str(labels),
                    "--classes",
                    "person",
                    "--out-dir",
                    str(output),
                    "--export-media",
                    mode,
                ]
            )
            == 0
        )
        plan = attach.build_attach_plan(
            HashDataset([(sample_id, digest)]),
            output,
            ["person"],
        )
        assert not [row for row in plan.issues if row["issue"] != "duplicate_task_image_hash"]
        assert plan.to_write[sample_id].boxes == (("person", (0.3, 0.3, 0.4, 0.4)),)


def test_label_field_defaults_and_can_be_selected():
    default_args = attach.parse_args(
        [
            "--dataset",
            "dataset",
            "--task-dir",
            ".",
            "--classes",
            "person",
            "--sample-tags",
            "batch",
        ]
    )
    custom_args = attach.parse_args(
        [
            "--dataset",
            "dataset",
            "--task-dir",
            ".",
            "--classes",
            "person",
            "--sample-tags",
            "batch",
            "--label-field",
            "ground_truth_detect",
        ]
    )
    assert default_args.label_field == "ground_truth"
    assert custom_args.label_field == "ground_truth_detect"


def test_existing_boxes_read_in_batches(monkeypatch):
    monkeypatch.setattr(attach, "QUERY_BATCH_SIZE", 2)
    calls = []

    class Dataset:
        def has_field(self, name):
            return True

        def select(self, ids):
            calls.append(ids)

            class View:
                def values(self, fields):
                    assert fields == ["id", attach.DEFAULT_LABEL_FIELD]
                    return ids, [None] * len(ids)

            return View()

    assert attach.load_existing_boxes(Dataset(), ["a", "b", "c"]) == dict.fromkeys(["a", "b", "c"])
    assert calls == [["a", "b"], ["c"]]


def test_export_xlabel_includes_sha256(tmp_path):
    export = load_script("export_xlabel.py")
    item = export.ExportItem(
        sample_id="abc",
        filepath=tmp_path / "a.jpg",
        relpath="a.jpg",
        json_relpath="a.json",
        width=10,
        height=20,
        shapes=[],
        sha256="deadbeef",
    )
    document = export.build_xlabel_document(item)
    assert document["sample_id"] == "abc"
    assert document["sha256"] == "deadbeef"
    assert document["imagePath"] == "a.jpg"
