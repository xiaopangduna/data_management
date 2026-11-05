"""42-point CVAT skeleton upload and pullback."""
import importlib.util
import math
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


shapes = load_script("cvat_keypoints")
export = load_script("export_cvat_keypoints")
update = load_script("update_cvat_keypoints")


def labels() -> list[str]:
    return [f"p{index}" for index in range(shapes.POINT_COUNT)]


def test_skeleton_label_keeps_point_order():
    label = shapes.skeleton_label(labels(), [[0, 1], [41, 40]])
    assert label["name"] == "kpt_face"
    assert label["type"] == "skeleton"
    assert [item["name"] for item in label["sublabels"]] == labels()
    assert not label["svg"].lstrip().startswith("<svg")
    assert 'data-type="element node"' in label["svg"]
    assert 'data-type="edge"' in label["svg"]
    assert 'data-node-from="41"' in label["svg"]
    assert 'data-label-name="p0"' in label["svg"]


def test_outside_point_round_trips_as_nan():
    points = [(0.25, 0.5)] * shapes.POINT_COUNT
    points[3] = (float("nan"), float("nan"))
    node_order = list(range(100, 142))
    shape = shapes.skeleton_shape(7, node_order, points, 200, 100, frame=2)
    assert shape["elements"][3]["outside"] is True
    assert shape["elements"][0]["points"] == [50, 50]
    faces = shapes.faces_from_shapes([shape], 2, 7, node_order, 200, 100)
    assert len(faces) == 1
    restored, visibility = faces[0]
    assert math.isnan(restored[3][0]) and visibility[3] == 0
    assert restored[0] == (0.25, 0.5) and visibility[0] == 1
    assert shapes.faces_from_shapes([shape], 0, 7, node_order, 200, 100) == []


def test_points_moved_ignores_pixel_rounding():
    before = [(0.10, 0.20)]
    after = [(0.11, 0.20)]
    assert shapes.points_moved(before, after, 100, 100, tolerance=5) is False
    assert shapes.points_moved(before, [(0.20, 0.20)], 100, 100, tolerance=5) is True
    assert shapes.points_moved(before, [(float("nan"), float("nan"))], 100, 100) is True


def test_project_contains_the_skeleton_and_its_box():
    skeleton = {"name": "kpt_face", "type": "skeleton", "sublabels": [{"name": "p0"}]}
    box = {"name": "kpt_face_box", "type": "rectangle"}
    assert shapes.project_sublabel_names([skeleton, box]) == ["p0"]
    assert shapes.project_sublabel_names([skeleton]) is None
    assert shapes.project_sublabel_names([skeleton, {"name": "head", "type": "rectangle"}]) is None


def test_rectangle_round_trips_and_pairs_by_group():
    shape = shapes.rectangle_shape(3, [0.1, 0.2, 0.3, 0.4], 100, 50, frame=1, group=2)
    assert shape["points"] == [10, 10, 40, 30]
    boxes = shapes.boxes_from_shapes([shape], 1, 3, 100, 50)
    assert boxes == [([0.1, 0.2, 0.3, 0.4], 2)]
    points = [(0.2, 0.2)]
    visibility = [1]
    paired = shapes.pair_downloaded([(points, visibility, 2)], boxes)
    assert paired == [([0.1, 0.2, 0.3, 0.4], points, visibility)]
    assert shapes.pair_downloaded([(points, visibility, 2)], []) == []
    with pytest.raises(ValueError, match="ungrouped"):
        shapes.pair_downloaded(
            [(points, visibility, 0), (points, visibility, 0)],
            [([0.1, 0.2, 0.1, 0.1], 0), ([0.2, 0.2, 0.1, 0.1], 0)],
        )


def face_sample(path: Path, label: str = "kpt_face") -> fo.Sample:
    points = [(0.1, 0.2)] * shapes.POINT_COUNT
    points[0] = (float("nan"), float("nan"))
    return fo.Sample(
        filepath=str(path),
        tags=["kpt_train"],
        metadata=fo.ImageMetadata(width=100, height=50),
        ground_truth=fo.Detections(detections=[
            fo.Detection(label="head", bounding_box=[0.0, 0.0, 0.1, 0.1]),
            fo.Detection(label=label, bounding_box=[0.2, 0.2, 0.3, 0.3]),
        ]),
        keypoints=fo.Keypoints(keypoints=[fo.Keypoint(label=label, points=points)]),
    )


def test_dry_run_reports_the_42_point_upload(tmp_path, capsys):
    image = tmp_path / "face.jpg"
    Image.new("RGB", (8, 8)).save(image)
    dataset = fo.Dataset()
    try:
        dataset.skeletons = {
            "keypoints": fo.KeypointSkeleton(labels=labels(), edges=[[0, 1]]),
        }
        dataset.save()
        dataset.add_sample(face_sample(image))
        args = export.parse_args([
            "--dataset", dataset.name,
            "--sample-tags", "kpt_train",
            "--anno-key", "cvat_kpt_42",
            "--dry-run",
        ])
        assert export.run(args) == 0
        output = capsys.readouterr().out
        assert "box_label=kpt_face_box" in output
        assert "pairs=1" in output
        assert shapes.RUNS_KEY not in (dataset.info or {})
    finally:
        dataset.delete()


def test_dry_run_accepts_another_kpt_label(tmp_path, capsys):
    image = tmp_path / "face.jpg"
    Image.new("RGB", (8, 8)).save(image)
    dataset = fo.Dataset()
    try:
        dataset.skeletons = {
            "keypoints": fo.KeypointSkeleton(labels=labels(), edges=[[0, 1]]),
        }
        dataset.save()
        dataset.add_sample(face_sample(image, "kpt_hand"))
        args = export.parse_args([
            "--dataset", dataset.name,
            "--sample-tags", "kpt_train",
            "--anno-key", "cvat_kpt_hand",
            "--label", "kpt_hand",
            "--dry-run",
        ])
        assert export.run(args) == 0
        assert "box_label=kpt_hand_box" in capsys.readouterr().out
    finally:
        dataset.delete()


def test_label_must_start_with_kpt():
    dataset = fo.Dataset()
    try:
        dataset.add_sample(fo.Sample(filepath="/tmp/face.jpg", tags=["kpt_train"]))
        args = export.parse_args([
            "--dataset", dataset.name,
            "--sample-tags", "kpt_train",
            "--anno-key", "cvat_kpt_42",
            "--label", "face",
            "--dry-run",
        ])
        with pytest.raises(ValueError, match="kpt"):
            export.run(args)
    finally:
        dataset.delete()


def test_unpaired_face_stops_the_upload(tmp_path):
    image = tmp_path / "face.jpg"
    Image.new("RGB", (8, 8)).save(image)
    dataset = fo.Dataset()
    try:
        dataset.skeletons = {
            "keypoints": fo.KeypointSkeleton(labels=labels(), edges=[[0, 1]]),
        }
        dataset.save()
        sample = face_sample(image)
        sample.ground_truth.detections.append(
            fo.Detection(label="kpt_face", bounding_box=[0.4, 0.4, 0.2, 0.2])
        )
        dataset.add_sample(sample)
        args = export.parse_args([
            "--dataset", dataset.name,
            "--sample-tags", "kpt_train",
            "--anno-key", "cvat_kpt_42",
            "--dry-run",
        ])
        with pytest.raises(ValueError, match="do not pair"):
            export.run(args)
    finally:
        dataset.delete()


def test_pullback_replaces_the_face_pair_and_keeps_indexes():
    dataset = fo.Dataset()
    try:
        other = fo.Keypoint(label="ear", points=[(0.3, 0.3)])
        original = [(0.2, 0.2)] * shapes.POINT_COUNT
        instance = fo.Instance()
        sample = fo.Sample(
            filepath="/tmp/face.jpg",
            tags=["train"],
            metadata=fo.ImageMetadata(width=100, height=100),
            ground_truth=fo.Detections(detections=[
                fo.Detection(label="head", bounding_box=[0.0, 0.0, 0.2, 0.2]),
                fo.Detection(
                    label="kpt_face",
                    bounding_box=[0.2, 0.2, 0.3, 0.3],
                    instance=instance,
                    tags=["keep"],
                ),
            ]),
            keypoints=fo.Keypoints(keypoints=[
                other,
                fo.Keypoint(label="kpt_face", points=original, instance=instance),
            ]),
        )
        dataset.add_sample(sample)
        moved = [(0.2, 0.2)] * shapes.POINT_COUNT
        moved[1] = (0.8, 0.8)
        moved[2] = (float("nan"), float("nan"))
        record = {
            "keypoints_field": "keypoints",
            "label_field": "ground_truth",
            "label": "kpt_face",
            "tasks": [{"sample_ids": [sample.id]}],
        }
        changed, unchanged = update.apply_pairs(
            dataset,
            record,
            {sample.id: [([0.5, 0.5, 0.2, 0.2], moved, [0 if index == 2 else 1 for index in range(shapes.POINT_COUNT)])]},
            {sample.id: (100, 100)},
            ["kpt_cvat"],
        )
        assert (changed, unchanged) == (1, 0)
        saved_sample = dataset[sample.id]
        saved = saved_sample.keypoints.keypoints
        boxes = saved_sample.ground_truth.detections
        assert [keypoint.label for keypoint in saved] == ["ear", "kpt_face"]
        assert [detection.label for detection in boxes] == ["head", "kpt_face"]
        assert list(boxes[1].bounding_box) == [0.5, 0.5, 0.2, 0.2]
        assert boxes[1].tags == ["keep"]
        assert boxes[1].instance.id == instance.id
        assert saved[1].instance.id == instance.id
        assert math.isnan(saved[1].points[2][0])
        assert list(saved[1].points[1]) == [0.8, 0.8]
        assert saved[1].visibility[2] == 0
        assert saved_sample.tags == ["train", "kpt_cvat", "changed"]
    finally:
        dataset.delete()


def test_pullback_drops_a_deleted_pair():
    dataset = fo.Dataset()
    try:
        points = [(0.2, 0.2)] * shapes.POINT_COUNT
        sample = fo.Sample(
            filepath="/tmp/face.jpg",
            tags=["train"],
            metadata=fo.ImageMetadata(width=100, height=100),
            ground_truth=fo.Detections(detections=[
                fo.Detection(label="head", bounding_box=[0.0, 0.0, 0.2, 0.2]),
                fo.Detection(label="kpt_face", bounding_box=[0.2, 0.2, 0.3, 0.3]),
            ]),
            keypoints=fo.Keypoints(keypoints=[
                fo.Keypoint(label="ear", points=[(0.3, 0.3)]),
                fo.Keypoint(label="kpt_face", points=points),
            ]),
        )
        dataset.add_sample(sample)
        record = {
            "keypoints_field": "keypoints",
            "label_field": "ground_truth",
            "label": "kpt_face",
            "tasks": [{"sample_ids": [sample.id]}],
        }
        changed, unchanged = update.apply_pairs(
            dataset, record, {sample.id: []}, {sample.id: (100, 100)}, ["kpt_cvat"]
        )
        assert (changed, unchanged) == (1, 0)
        saved = dataset[sample.id]
        assert [detection.label for detection in saved.ground_truth.detections] == ["head"]
        assert [keypoint.label for keypoint in saved.keypoints.keypoints] == ["ear"]
    finally:
        dataset.delete()


def test_pullback_dry_run_reports_changes_without_writing(tmp_path, capsys, monkeypatch):
    del tmp_path
    dataset = fo.Dataset()
    try:
        points = [(0.2, 0.2)] * shapes.POINT_COUNT
        dataset.add_sample(fo.Sample(
            filepath="/tmp/face.jpg",
            tags=["train"],
            metadata=fo.ImageMetadata(width=10, height=10),
            ground_truth=fo.Detections(detections=[
                fo.Detection(label="kpt_face", bounding_box=[0.2, 0.2, 0.3, 0.3]),
            ]),
            keypoints=fo.Keypoints(keypoints=[
                fo.Keypoint(label="kpt_face", points=points),
            ]),
        ))
        sample_id = dataset.first().id
        export.save_run(dataset, "cvat_kpt_42", {
            "project_name": "BBM08S_head_keypoints_42",
            "project_id": 1,
            "keypoints_field": "keypoints",
            "label_field": "ground_truth",
            "label": "kpt_face",
            "box_label": "kpt_face_box",
            "labels": labels(),
            "tasks": [{"task_id": 9, "sample_ids": [sample_id]}],
        })
        monkeypatch.setattr(update, "connect", lambda api: object())
        monkeypatch.setattr(update, "downloaded_faces", lambda *args, **kwargs: {sample_id: []})
        args = update.parse_args([
            "--dataset", dataset.name,
            "--anno-key", "cvat_kpt_42",
            "--sample-tags", "kpt_cvat",
            "--dry-run",
        ])
        assert update.run(args) == 0
        output = capsys.readouterr().out
        assert "changed=1" in output
        assert "unchanged=0" in output
        saved = dataset[sample_id]
        assert [detection.label for detection in saved.ground_truth.detections] == ["kpt_face"]
        assert saved.tags == ["train"]
    finally:
        dataset.delete()


def test_rejects_a_68_point_field():
    dataset = fo.Dataset()
    try:
        dataset.skeletons = {
            "keypoints_68": fo.KeypointSkeleton(labels=[f"f{index}" for index in range(68)], edges=[[0, 1]]),
        }
        dataset.save()
        dataset.add_sample(fo.Sample(filepath="/tmp/face.jpg", tags=["kpt68_train"]))
        args = export.parse_args([
            "--dataset", dataset.name,
            "--sample-tags", "kpt68_train",
            "--anno-key", "cvat_kpt_68",
            "--keypoints-field", "keypoints_68",
            "--dry-run",
        ])
        with pytest.raises(ValueError, match="42"):
            export.run(args)
    finally:
        dataset.delete()
