"""Export paired face keypoints to a COCO JSON for training."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import fiftyone as fo
import pytest
from PIL import Image

ROOT = Path(__file__).parents[1] / "scripts"
spec = importlib.util.spec_from_file_location(
    "export_coco_keypoints", ROOT / "export_coco_keypoints.py"
)
export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export
spec.loader.exec_module(export)

shapes_spec = importlib.util.spec_from_file_location("cvat_keypoints", ROOT / "cvat_keypoints.py")
shapes = importlib.util.module_from_spec(shapes_spec)
sys.modules[shapes_spec.name] = shapes
shapes_spec.loader.exec_module(shapes)


def names(count: int) -> list[str]:
    return [f"p{index}" for index in range(count)]


def save_image(path: Path, size: tuple[int, int] = (100, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "red").save(path)


def face_points(count: int) -> tuple[list[tuple[float, float]], list[int]]:
    points = [(0.25, 0.5)] * count
    visibility = [2] * count
    points[0] = (float("nan"), float("nan"))
    visibility[0] = 0
    points[1] = (0.1, 0.2)
    visibility[1] = 1
    return points, visibility


def face_keypoint(count: int, instance: fo.Instance | None = None) -> fo.Keypoint:
    points, visibility = face_points(count)
    keypoint = fo.Keypoint(label="kpt_face", points=points, instance=instance)
    keypoint["visibility"] = visibility
    return keypoint


def add_visibility_field(dataset: fo.Dataset, field_name: str) -> None:
    if not dataset.has_field(field_name):
        dataset.add_sample_field(field_name, fo.EmbeddedDocumentField, embedded_doc_type=fo.Keypoints)
    dataset.add_sample_field(
        f"{field_name}.keypoints.visibility", fo.ListField, subfield=fo.IntField,
    )


def test_parse_args_defaults(tmp_path):
    args = export.parse_args([
        "--dataset", "demo",
        "--out-dir", str(tmp_path),
        "--sample-tags", "kpt_train",
        "--keypoints-field", "keypoints",
    ])
    assert args.split == "train"
    assert args.label == "kpt_face"
    assert args.label_field == "ground_truth"
    assert args.exclude_tags == ["dup_repeat_drop", "dup_near_drop"]
    assert args.export_media == "symlink"
    with pytest.raises(SystemExit):
        export.parse_args([
            "--dataset", "demo",
            "--out-dir", str(tmp_path),
            "--sample-tags", "kpt_train",
            "--keypoints-field", "poses",
        ])


def test_coco_triple_keeps_occluded_points_and_zeros_unlabeled():
    assert export.coco_triple((0.25, 0.5), 2, 100, 40) == (25.0, 20.0, 2)
    assert export.coco_triple((0.1, 0.2), 1, 100, 40) == (10.0, 8.0, 1)
    assert export.coco_triple((0.1, 0.2), 0, 100, 40) == (0.0, 0.0, 0)
    assert export.coco_triple((float("nan"), float("nan")), 2, 100, 40) == (0.0, 0.0, 0)
    assert export.pixel_box([0.2, 0.25, 0.5, 0.5], 100, 40) == [20.0, 10.0, 50.0, 20.0]
    assert export.pixel_box([0.0, 0.0, 0.0, 0.2], 100, 40) is None


def test_check_output_rejects_existing_split(tmp_path):
    output = tmp_path / "out"
    images = output / "images"
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="empty or absent"):
        export.check_output(output, "train")
    empty = tmp_path / "empty"
    (empty / "images").mkdir(parents=True)
    (empty / "instance_train.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        export.check_output(empty, "train")


def test_export_writes_coco_faces_and_skips_bad_samples(tmp_path):
    face = tmp_path / "src" / "face.jpg"
    two = tmp_path / "src" / "two.jpg"
    none = tmp_path / "src" / "none.jpg"
    nosize = tmp_path / "src" / "nosize.jpg"
    unpaired = tmp_path / "src" / "unpaired.jpg"
    dropped = tmp_path / "src" / "dropped.jpg"
    for path in (face, two, none, nosize, unpaired, dropped):
        save_image(path)
    dataset = fo.Dataset()
    try:
        dataset.skeletons = {
            "keypoints": fo.KeypointSkeleton(labels=names(shapes.POINT_COUNT), edges=[[0, 1], [2, 3]]),
        }
        dataset.save()
        add_visibility_field(dataset, "keypoints")
        first = fo.Instance()
        second = fo.Instance()
        dataset.add_samples([
            fo.Sample(
                filepath=str(face),
                tags=["kpt_train"],
                metadata=fo.ImageMetadata(width=100, height=40),
                ground_truth=fo.Detections(detections=[
                    fo.Detection(label="head", bounding_box=[0.0, 0.0, 0.05, 0.05]),
                    fo.Detection(label="kpt_face", bounding_box=[0.2, 0.25, 0.5, 0.5]),
                ]),
                keypoints=fo.Keypoints(keypoints=[face_keypoint(shapes.POINT_COUNT)]),
            ),
            fo.Sample(
                filepath=str(two),
                tags=["kpt_train"],
                metadata=fo.ImageMetadata(width=100, height=40),
                ground_truth=fo.Detections(detections=[
                    fo.Detection(label="head", bounding_box=[0.0, 0.0, 0.05, 0.05]),
                    fo.Detection(label="kpt_face", bounding_box=[0.1, 0.1, 0.2, 0.2], instance=first),
                    fo.Detection(label="kpt_face", bounding_box=[0.4, 0.4, 0.2, 0.3], instance=second),
                ]),
                keypoints=fo.Keypoints(keypoints=[
                    face_keypoint(shapes.POINT_COUNT, first),
                    face_keypoint(shapes.POINT_COUNT, second),
                ]),
            ),
            fo.Sample(
                filepath=str(none),
                tags=["kpt_train"],
                metadata=fo.ImageMetadata(width=100, height=40),
                ground_truth=fo.Detections(detections=[
                    fo.Detection(label="head", bounding_box=[0.0, 0.0, 0.2, 0.2]),
                ]),
            ),
            fo.Sample(
                filepath=str(nosize),
                tags=["kpt_train"],
                ground_truth=fo.Detections(detections=[
                    fo.Detection(label="kpt_face", bounding_box=[0.2, 0.2, 0.2, 0.2]),
                ]),
                keypoints=fo.Keypoints(keypoints=[face_keypoint(shapes.POINT_COUNT)]),
            ),
            fo.Sample(
                filepath=str(unpaired),
                tags=["kpt_train"],
                metadata=fo.ImageMetadata(width=100, height=40),
                ground_truth=fo.Detections(detections=[
                    fo.Detection(label="kpt_face", bounding_box=[0.1, 0.1, 0.2, 0.2], instance=fo.Instance()),
                    fo.Detection(label="kpt_face", bounding_box=[0.4, 0.4, 0.2, 0.2], instance=fo.Instance()),
                ]),
                keypoints=fo.Keypoints(keypoints=[face_keypoint(shapes.POINT_COUNT)]),
            ),
            fo.Sample(
                filepath=str(dropped),
                tags=["kpt_train", "dup_near_drop"],
                metadata=fo.ImageMetadata(width=100, height=40),
                ground_truth=fo.Detections(detections=[
                    fo.Detection(label="kpt_face", bounding_box=[0.2, 0.2, 0.2, 0.2]),
                ]),
                keypoints=fo.Keypoints(keypoints=[face_keypoint(shapes.POINT_COUNT)]),
            ),
        ])
        output = tmp_path / "out"
        args = export.parse_args([
            "--dataset", dataset.name,
            "--out-dir", str(output),
            "--sample-tags", "kpt_train",
            "--keypoints-field", "keypoints",
            "--dry-run",
        ])
        dry = export.export_dataset(args, report_dir=tmp_path)
        assert dry["images"] == 2 and dry["faces"] == 3 and dry["skipped"] == 3
        assert dry["view_samples"] == 5
        assert dry["skip_counts"] == {"no_face": 1, "missing_size": 1, "unpaired": 1}
        assert not (output / "images").exists()
        assert not (output / "instance_train.json").exists()

        written = export.export_dataset(
            export.parse_args([
                "--dataset", dataset.name,
                "--out-dir", str(output),
                "--sample-tags", "kpt_train",
                "--keypoints-field", "keypoints",
            ]),
            report_dir=tmp_path,
        )
        assert written["point_count"] == 42
        document = json.loads((output / "instance_train.json").read_text(encoding="utf-8"))
        assert document["categories"][0]["keypoints"] == names(42)
        assert document["categories"][0]["skeleton"] == [[0, 1], [2, 3]]
        by_name = {image["file_name"]: image for image in document["images"]}
        assert set(by_name) == {"face.jpg", "two.jpg"}
        face_image = by_name["face.jpg"]
        face_anns = [ann for ann in document["annotations"] if ann["image_id"] == face_image["id"]]
        assert len(face_anns) == 1
        ann = face_anns[0]
        assert ann["bbox"] == ann["face_box"] == [20.0, 10.0, 50.0, 20.0]
        assert ann["num_keypoints"] == 41
        assert ann["keypoints"][0:3] == [0.0, 0.0, 0]
        assert ann["keypoints"][3:6] == [10.0, 8.0, 1]
        assert ann["keypoints"][6:9] == [25.0, 20.0, 2]
        assert len(ann["keypoints"]) == 126
        two_image = by_name["two.jpg"]
        two_anns = [ann for ann in document["annotations"] if ann["image_id"] == two_image["id"]]
        assert [ann["bbox"] for ann in two_anns] == [
            [10.0, 4.0, 20.0, 8.0],
            [40.0, 16.0, 20.0, 12.0],
        ]
        link = output / "images" / "face.jpg"
        assert link.is_symlink()
        assert link.read_bytes() == face.read_bytes()
        with (tmp_path / f"export_coco_keypoints_{dataset.name}.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["issue"] for row in rows} == {"no_face", "missing_size", "unpaired"}
        assert not any(row["filepath"].endswith("dropped.jpg") for row in rows)
    finally:
        dataset.delete()


def test_duplicate_names_stop_before_writing(tmp_path):
    first = tmp_path / "a" / "face.jpg"
    second = tmp_path / "b" / "face.jpg"
    save_image(first)
    save_image(second)
    dataset = fo.Dataset()
    try:
        dataset.skeletons = {
            "keypoints": fo.KeypointSkeleton(labels=names(shapes.POINT_COUNT), edges=[[0, 1]]),
        }
        dataset.save()
        add_visibility_field(dataset, "keypoints")
        for path in (first, second):
            dataset.add_sample(fo.Sample(
                filepath=str(path),
                tags=["kpt_train"],
                metadata=fo.ImageMetadata(width=100, height=40),
                ground_truth=fo.Detections(detections=[
                    fo.Detection(label="kpt_face", bounding_box=[0.2, 0.2, 0.3, 0.3]),
                ]),
                keypoints=fo.Keypoints(keypoints=[face_keypoint(shapes.POINT_COUNT)]),
            ))
        output = tmp_path / "out"
        args = export.parse_args([
            "--dataset", dataset.name,
            "--out-dir", str(output),
            "--sample-tags", "kpt_train",
            "--keypoints-field", "keypoints",
        ])
        with pytest.raises(ValueError, match="Duplicate image names"):
            export.export_dataset(args, report_dir=tmp_path)
        assert not (output / "instance_train.json").exists()
        assert not (output / "images").exists()
    finally:
        dataset.delete()


def test_export_68_points_uses_split_name_and_copy(tmp_path):
    source = tmp_path / "src" / "face68.jpg"
    save_image(source, (80, 80))
    dataset = fo.Dataset()
    try:
        dataset.skeletons = {
            "keypoints_68": fo.KeypointSkeleton(
                labels=names(shapes.POINT_COUNT_68), edges=[[0, 67]],
            ),
        }
        dataset.save()
        add_visibility_field(dataset, "keypoints_68")
        points = [(0.5, 0.5)] * shapes.POINT_COUNT_68
        visibility = [2] * shapes.POINT_COUNT_68
        keypoint = fo.Keypoint(label="kpt_face", points=points)
        keypoint["visibility"] = visibility
        dataset.add_sample(fo.Sample(
            filepath=str(source),
            tags=["kpt_val"],
            metadata=fo.ImageMetadata(width=80, height=80),
            ground_truth=fo.Detections(detections=[
                fo.Detection(label="kpt_face", bounding_box=[0.25, 0.25, 0.5, 0.5]),
            ]),
            keypoints_68=fo.Keypoints(keypoints=[keypoint]),
        ))
        output = tmp_path / "val"
        args = export.parse_args([
            "--dataset", dataset.name,
            "--out-dir", str(output),
            "--split", "val",
            "--sample-tags", "kpt_val",
            "--keypoints-field", "keypoints_68",
            "--export-media", "copy",
        ])
        summary = export.export_dataset(args, report_dir=tmp_path)
        assert summary["point_count"] == 68 and summary["faces"] == 1
        document = json.loads((output / "instance_val.json").read_text(encoding="utf-8"))
        assert len(document["categories"][0]["keypoints"]) == 68
        assert len(document["annotations"][0]["keypoints"]) == 204
        assert document["annotations"][0]["face_box"] == [20.0, 20.0, 40.0, 40.0]
        image = output / "images" / "face68.jpg"
        assert image.is_file() and not image.is_symlink()
        assert image.read_bytes() == source.read_bytes()
    finally:
        dataset.delete()
