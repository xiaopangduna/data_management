"""Attach COCO face boxes and 42 keypoints onto existing samples."""
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import fiftyone as fo
import pytest
from PIL import Image

ROOT = Path(__file__).parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


update = load_script("update_face_keypoints")


def names() -> list[str]:
    return [f"p{index}" for index in range(update.POINT_COUNT)]


def keypoints(width: int, height: int, hidden: int = 0) -> list[float]:
    flat: list[float] = []
    for index in range(update.POINT_COUNT):
        flat.extend([float(index), float(index), 0 if index == hidden else 1])
    return flat


def write_coco(path: Path, filename: str, width: int = 100, height: int = 50, **changes) -> None:
    image = {
        "id": 1,
        "file_name": filename,
        "width": width,
        "height": height,
    }
    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [0, 0, 10, 10],
        "face_box": [10, 20, 30, 40],
        "keypoints": keypoints(width, height),
        "num_keypoints": update.POINT_COUNT,
    }
    image.update(changes.pop("image", {}))
    annotation.update(changes.pop("annotation", {}))
    document = {
        "images": changes.pop("images", [image]),
        "annotations": changes.pop("annotations", [annotation]),
        "categories": [{
            "id": 1,
            "name": "face",
            "keypoints": names(),
            "skeleton": [[0, 1], [1, 2]],
        }],
    }
    document.update(changes)
    path.write_text(json.dumps(document), encoding="utf-8")


def add_image(directory: Path, name: str, size: tuple[int, int] = (8, 8)) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.new("RGB", size).save(path)
    return path


def copy_image(source: Path, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    shutil.copyfile(source, path)
    return path


def hashed_sample(path: Path, tags: list[str], **fields) -> fo.Sample:
    sample = fo.Sample(filepath=str(path), tags=tags, sha256=update.compute_sha256_hex(path))
    for name, value in fields.items():
        sample[name] = value
    return sample


def test_parse_args_requires_paths_and_tags():
    with pytest.raises(SystemExit):
        update.parse_args(["--dataset", "BBM08S_head"])
    args = update.parse_args([
        "--dataset", "BBM08S_head",
        "--image-dir", "images",
        "--json", "instance_train.json",
        "--sample-tags", "kpt_train,review",
        "--dry-run",
    ])
    assert args.sample_tags == ["kpt_train", "review"]
    assert args.label == "kpt_face"
    assert args.label_field == "ground_truth"
    assert args.keypoints_field == "keypoints"
    assert args.dry_run is True
    with pytest.raises(SystemExit):
        update.parse_args([
            "--dataset", "BBM08S_head",
            "--image-dir", "images",
            "--json", "instance_train.json",
            "--sample-tags", "kpt_train",
            "--label-field", "keypoints",
            "--keypoints-field", "keypoints",
        ])


def test_parse_coco_hides_missing_points_and_ignores_tight_bbox(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    add_image(image_dir, "a.jpg")
    write_coco(tmp_path / "one.json", "a.jpg")
    document = json.loads((tmp_path / "one.json").read_text(encoding="utf-8"))
    schema, labels, issues, count = update.parse_coco(document, image_dir)
    assert count == 1 and issues == []
    assert schema.labels[0] == "p0"
    assert schema.edges == ((0, 1), (1, 2))
    face = labels["a.jpg"]
    assert face.box == (0.1, 0.4, 0.3, 0.8)
    assert math.isnan(face.points[0][0]) and math.isnan(face.points[0][1])
    assert face.points[1] == (0.01, 0.02)
    assert face.visibility[0] == 0 and face.visibility[1] == 1
    assert len(face.points) == update.POINT_COUNT


def test_bbox_is_the_fallback_when_face_box_is_missing(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    add_image(image_dir, "a.jpg")
    write_coco(tmp_path / "one.json", "a.jpg", annotation={"face_box": None})
    document = json.loads((tmp_path / "one.json").read_text(encoding="utf-8"))
    _schema, labels, issues, _count = update.parse_coco(document, image_dir)
    assert issues == []
    assert labels["a.jpg"].box == (0.0, 0.0, 0.1, 0.2)


def test_face_box_wins_when_both_boxes_exist(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    add_image(image_dir, "a.jpg")
    write_coco(tmp_path / "one.json", "a.jpg")
    document = json.loads((tmp_path / "one.json").read_text(encoding="utf-8"))
    _schema, labels, issues, _count = update.parse_coco(document, image_dir)
    assert issues == []
    assert labels["a.jpg"].box == (0.1, 0.4, 0.3, 0.8)


def test_68_points_use_their_own_field_and_keep_coordinates(tmp_path):
    image_dir = tmp_path / "source"
    source = add_image(image_dir, "face.png")
    stored = copy_image(source, tmp_path / "stored", "library.png")
    flat = []
    for index in range(update.POINT_COUNT_68):
        flat.extend([float(index + 1), float(index + 1), 0 if index == 0 else 4 if index == 1 else 1])
    write_coco(
        tmp_path / "one.json",
        "face.png",
        annotation={"keypoints": flat, "num_keypoints": update.POINT_COUNT_68, "face_box": None},
        categories=[{
            "id": 1,
            "name": "face",
            "keypoints": [f"f{index}" for index in range(update.POINT_COUNT_68)],
            "skeleton": [[0, 1], [35, 36], [67, 60]],
        }],
    )
    dataset = fo.Dataset()
    try:
        sample = hashed_sample(
            stored,
            ["train"],
            ground_truth=fo.Detections(detections=[
                fo.Detection(label="kpt_face", bounding_box=[0.2, 0.2, 0.2, 0.2])
            ]),
            keypoints=fo.Keypoints(keypoints=[fo.Keypoint(label="kpt_face", points=[(0.1, 0.1)])]),
        )
        dataset.add_sample(sample)
        result = update.update_dataset(
            dataset, image_dir, tmp_path / "one.json", ["kpt68_train"], report_dir=tmp_path,
        )
        assert (result.matched, result.written, result.issues) == (1, 1, [])
        saved = dataset[sample.id]
        assert [detection.label for detection in saved.ground_truth.detections] == ["kpt_face"]
        assert saved.ground_truth.detections[0].bounding_box == [0.2, 0.2, 0.2, 0.2]
        assert saved.keypoints.keypoints[0].label == "kpt_face"
        face = saved.keypoints_68.keypoints[0]
        assert face.label == "kpt_face"
        assert len(face.points) == update.POINT_COUNT_68
        assert face.points[0] == [0.01, 0.02]
        assert face.visibility[0] == 0 and face.visibility[1] == 4
        assert list(dataset.skeletons["keypoints_68"].labels) == [f"f{index}" for index in range(68)]
        assert "keypoints" not in dataset.skeletons
    finally:
        dataset.delete()


def test_parse_coco_rejects_a_different_point_count():
    document = {
        "images": [],
        "annotations": [],
        "categories": [{"keypoints": ["p0"], "skeleton": [[0, 0]]}],
    }
    with pytest.raises(ValueError, match="42"):
        update.parse_coco(document, Path("."))


def test_parse_coco_reports_a_missing_file(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    write_coco(tmp_path / "one.json", "gone.jpg")
    document = json.loads((tmp_path / "one.json").read_text(encoding="utf-8"))
    _schema, labels, issues, _count = update.parse_coco(document, image_dir)
    assert labels == {}
    assert issues[0]["issue"] == "missing_file"
    assert issues[0]["filename"] == "gone.jpg"


def test_merge_tags_appends_without_replacing_or_duplicating():
    assert update.merge_tags(["train", "train_v001"], ["kpt_train"]) == (
        ["train", "train_v001", "kpt_train"],
        True,
    )
    assert update.merge_tags(["kpt_train", "train"], ["kpt_train"]) == (
        ["kpt_train", "train"],
        False,
    )


def test_update_matches_sha256_and_keeps_existing_boxes(tmp_path):
    image_dir = tmp_path / "source"
    source = add_image(image_dir, "json_name.jpg")
    stored_path = copy_image(source, tmp_path / "stored", "library_name.jpg")
    write_coco(tmp_path / "one.json", "json_name.jpg")
    dataset = fo.Dataset()
    try:
        sample = hashed_sample(
            stored_path,
            ["train"],
            ground_truth=fo.Detections(detections=[
                fo.Detection(label="head", bounding_box=[0.2, 0.2, 0.2, 0.2])
            ]),
        )
        dataset.add_sample(sample)
        result = update.update_dataset(
            dataset,
            image_dir,
            tmp_path / "one.json",
            ["kpt_train"],
            report_dir=tmp_path,
        )
        assert (result.matched, result.to_update, result.written, result.issues) == (1, 1, 1, [])
        saved = dataset[sample.id]
        assert saved.filepath == str(stored_path)
        assert saved.tags == ["train", "kpt_train"]
        labels = [detection.label for detection in saved.ground_truth.detections]
        assert labels == ["head", "kpt_face"]
        face = saved.ground_truth.detections[1]
        assert face.bounding_box == [0.1, 0.4, 0.3, 0.8]
        keypoint = saved.keypoints.keypoints[0]
        assert keypoint.label == "kpt_face"
        assert len(keypoint.points) == update.POINT_COUNT
        assert math.isnan(keypoint.points[0][0])
        assert list(keypoint.visibility) == [0 if index == 0 else 1 for index in range(update.POINT_COUNT)]
        assert str(face.instance.id) == str(keypoint.instance.id)
        assert list(dataset.skeletons["keypoints"].labels) == names()
        assert [list(edge) for edge in dataset.skeletons["keypoints"].edges] == [[0, 1], [1, 2]]

        again = update.update_dataset(
            dataset,
            image_dir,
            tmp_path / "one.json",
            ["kpt_train"],
            report_dir=tmp_path,
        )
        assert (again.to_update, again.unchanged, again.written) == (0, 1, 0)
        assert [detection.label for detection in dataset[sample.id].ground_truth.detections] == [
            "head",
            "kpt_face",
        ]
    finally:
        dataset.delete()


def test_dry_run_does_not_write(tmp_path):
    image_dir = tmp_path / "source"
    source = add_image(image_dir, "a.jpg")
    stored = copy_image(source, tmp_path / "stored", "other.jpg")
    write_coco(tmp_path / "one.json", "a.jpg")
    dataset = fo.Dataset()
    try:
        sample_id = dataset.add_sample(hashed_sample(
            stored,
            ["train"],
            ground_truth=fo.Detections(detections=[
                fo.Detection(label="head", bounding_box=[0.1, 0.1, 0.1, 0.1])
            ]),
        ))
        result = update.update_dataset(
            dataset, image_dir, tmp_path / "one.json", ["kpt_val"], dry_run=True, report_dir=tmp_path,
        )
        assert result.to_update == 1 and result.written == 0
        saved = dataset[sample_id]
        assert saved.tags == ["train"]
        assert [detection.label for detection in saved.ground_truth.detections] == ["head"]
        assert dataset.has_field("keypoints") is False
    finally:
        dataset.delete()


def test_reports_samples_missing_from_the_dataset(tmp_path):
    image_dir = tmp_path / "source"
    image_dir.mkdir()
    add_image(image_dir, "a.jpg")
    write_coco(tmp_path / "one.json", "a.jpg")
    dataset = fo.Dataset()
    dataset.add_sample_field("sha256", fo.StringField)
    try:
        result = update.update_dataset(
            dataset, image_dir, tmp_path / "one.json", ["kpt_test"], report_dir=tmp_path,
        )
        assert result.matched == 0
        assert result.issues[0]["issue"] == "not_in_dataset"
        assert Path(result.csv_path).is_file()
        assert dataset.has_field("keypoints") is False
    finally:
        dataset.delete()


def test_metadata_size_is_used_when_coordinates_fit(tmp_path):
    image_dir = tmp_path / "source"
    source = add_image(image_dir, "a.jpg")
    stored = copy_image(source, tmp_path / "stored", "other.jpg")
    write_coco(
        tmp_path / "one.json",
        "a.jpg",
        width=200,
        height=100,
        annotation={"face_box": [10, 5, 20, 10]},
    )
    dataset = fo.Dataset()
    try:
        sample = hashed_sample(stored, ["train"], metadata=fo.ImageMetadata(width=100, height=50))
        dataset.add_sample(sample)
        result = update.update_dataset(
            dataset, image_dir, tmp_path / "one.json", ["kpt_train"], report_dir=tmp_path,
        )
        assert result.to_update == 1 and result.issues == []
        face = dataset[sample.id].ground_truth.detections[0]
        assert face.bounding_box == [0.1, 0.1, 0.2, 0.2]
        assert list(dataset[sample.id].keypoints.keypoints[0].points[1]) == [0.01, 0.02]
    finally:
        dataset.delete()


def test_size_mismatch_skips_the_sample(tmp_path):
    image_dir = tmp_path / "source"
    source = add_image(image_dir, "a.jpg")
    stored = copy_image(source, tmp_path / "stored", "other.jpg")
    write_coco(tmp_path / "one.json", "a.jpg", width=100, height=50)
    dataset = fo.Dataset()
    try:
        sample = hashed_sample(stored, ["train"], metadata=fo.ImageMetadata(width=90, height=50))
        dataset.add_sample(sample)
        result = update.update_dataset(
            dataset, image_dir, tmp_path / "one.json", ["kpt_train"], report_dir=tmp_path,
        )
        assert result.matched == 1 and result.to_update == 0
        assert result.issues[0]["issue"] == "size_mismatch"
        assert dataset.has_field("ground_truth") is False
    finally:
        dataset.delete()


def test_exact_phash_matches_a_recompressed_image(tmp_path):
    image_dir = tmp_path / "source"
    image_dir.mkdir()
    stored = image_dir / "library.png"
    variant = image_dir / "json_name.png"
    image = Image.new("RGB", (32, 32), (12, 34, 56))
    image.save(stored, compress_level=9)
    image.save(variant, compress_level=0)
    assert update.compute_sha256_hex(stored) != update.compute_sha256_hex(variant)
    phash = update.compute_phash_hex(stored)
    assert phash == update.compute_phash_hex(variant)
    write_coco(tmp_path / "one.json", "json_name.png")
    dataset = fo.Dataset()
    try:
        sample = hashed_sample(stored, ["train"], phash=phash)
        dataset.add_sample(sample)
        result = update.update_dataset(
            dataset, image_dir, tmp_path / "one.json", ["kpt_train"], report_dir=tmp_path,
        )
        assert result.phash_matched == 1
        assert (result.matched, result.written, result.issues) == (1, 1, [])
        assert dataset[sample.id].tags == ["train", "kpt_train"]
    finally:
        dataset.delete()


def test_ambiguous_phash_is_not_updated(tmp_path):
    image_dir = tmp_path / "source"
    image_dir.mkdir()
    incoming = add_image(image_dir, "json_name.png", size=(32, 32))
    phash = update.compute_phash_hex(incoming)
    write_coco(tmp_path / "one.json", "json_name.png")
    dataset = fo.Dataset()
    try:
        first = add_image(tmp_path / "one", "a.png", size=(16, 16))
        second = add_image(tmp_path / "two", "b.png", size=(16, 16))
        dataset.add_samples([
            hashed_sample(first, ["train"], phash=phash),
            hashed_sample(second, ["train"], phash=phash),
        ])
        result = update.update_dataset(
            dataset, image_dir, tmp_path / "one.json", ["kpt_train"], report_dir=tmp_path,
        )
        assert result.matched == 0 and result.phash_matched == 0
        assert result.issues[0]["issue"] == "ambiguous_phash"
        assert dataset.has_field("keypoints") is False
    finally:
        dataset.delete()


def test_ambiguous_sha256_is_not_updated(tmp_path):
    image_dir = tmp_path / "source"
    source = add_image(image_dir, "a.jpg")
    write_coco(tmp_path / "one.json", "a.jpg")
    dataset = fo.Dataset()
    try:
        first = copy_image(source, tmp_path / "one", "library_a.jpg")
        second = copy_image(source, tmp_path / "two", "library_b.jpg")
        dataset.add_samples([
            hashed_sample(first, ["train"]),
            hashed_sample(second, ["train"]),
        ])
        result = update.update_dataset(
            dataset, image_dir, tmp_path / "one.json", ["kpt_train"], report_dir=tmp_path,
        )
        assert result.matched == 0
        assert result.issues[0]["issue"] == "ambiguous_sha256"
        assert dataset.has_field("keypoints") is False
    finally:
        dataset.delete()
