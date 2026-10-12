"""CVAT publish selection and class-scoped load."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


cvat = load_script("cvat")
cvat_load = load_script("cvat_load")


def test_publish_cli_requires_core_options():
    args = cvat.parse_args(
        [
            "--dataset", "BBM08S_head",
            "--sample-tags", "relabel,train",
            "--anno-key", "cvat_head_261012",
            "--classes", "baby_head,adult_head,baby_head",
        ]
    )
    assert args.sample_tags == ["relabel", "train"]
    assert args.classes == ["baby_head", "adult_head"]
    assert args.project_name is None
    assert args.label_field == "ground_truth"
    assert not args.dry_run
    assert cvat.resolve_project_name(args.dataset, args.project_name) == "BBM08S_head"
    assert cvat.pending_tag(args.anno_key) == "cvat_cvat_head_261012"
    with pytest.raises(SystemExit):
        cvat.parse_args(["--dataset", "BBM08S_head", "--sample-tags", "relabel", "--anno-key", "ok"])
    with pytest.raises(SystemExit):
        cvat.parse_args(
            [
                "--dataset", "BBM08S_head",
                "--sample-tags", "relabel",
                "--anno-key", "has space",
                "--classes", "baby_head",
            ]
        )


def test_task_size_stays_within_upload_budget():
    assert cvat.task_size_for_budget([100, 100, 100], budget=250) == 2
    assert cvat.task_count(3, 2) == 2
    assert cvat.task_size_for_budget([800, 100, 800], budget=900) == 2
    assert cvat.task_size_for_budget([10, 10, 10], budget=100) == 3
    assert cvat.task_size_for_budget([10] * 10, budget=10_000, max_files=4) == 4
    with pytest.raises(ValueError, match="above the 50 byte"):
        cvat.task_size_for_budget([80, 10], budget=50)


def test_project_classes_must_match_when_project_exists():
    cvat.ensure_project_classes(None, ["baby_head"])
    cvat.ensure_project_classes({"adult_head", "baby_head"}, ["baby_head", "adult_head"])
    with pytest.raises(ValueError, match="differ from --classes"):
        cvat.ensure_project_classes({"baby_head", "person"}, ["baby_head"])


def test_select_view_keeps_samples_with_every_tag(tmp_path):
    dataset = fo.Dataset()
    try:
        def add(name, tags):
            path = tmp_path / name
            Image.new("RGB", (8, 8)).save(path)
            dataset.add_sample(
                fo.Sample(
                    filepath=str(path),
                    tags=tags,
                    ground_truth=fo.Detections(),
                )
            )

        add("keep.jpg", ["relabel", "train"])
        add("partial.jpg", ["relabel"])
        add("also.jpg", ["relabel", "train", "dup_repeat_drop"])
        view = cvat.select_view(dataset, ["relabel", "train"])
        assert len(view) == 2
    finally:
        dataset.delete()


def test_dry_run_prints_plan_without_tagging(tmp_path, capsys):
    dataset_name = "cvat_publish_dry_run"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)
    try:
        path = tmp_path / "keep.jpg"
        Image.new("RGB", (8, 8)).save(path)
        dataset.add_sample(
            fo.Sample(filepath=str(path), tags=["relabel"], ground_truth=fo.Detections())
        )
        dataset.persistent = True
        args = cvat.parse_args(
            [
                "--dataset", dataset_name,
                "--sample-tags", "relabel",
                "--anno-key", "cvat_head_261012",
                "--classes", "baby_head",
                "--dry-run",
            ]
        )
        assert cvat.run(args) == 0
        output = capsys.readouterr().out
        assert "mode=dry-run" in output
        assert "samples=1" in output
        assert "tasks=1" in output
        assert "task_size=1" in output
        assert "project_name=cvat_publish_dry_run" in output
        assert dataset.first().tags == ["relabel"]
    finally:
        fo.delete_dataset(dataset_name)


def test_load_merges_only_requested_classes():
    names = {"baby_head"}
    before = fo.Detections(detections=[
        fo.Detection(label="adult_head", bounding_box=[0.1, 0.1, 0.2, 0.2]),
        fo.Detection(label="baby_head", bounding_box=[0.2, 0.2, 0.2, 0.2]),
    ])
    after = fo.Detections(detections=[
        fo.Detection(label="adult_head", bounding_box=[0.9, 0.9, 0.1, 0.1]),
        fo.Detection(label="baby_head", bounding_box=[0.6, 0.6, 0.2, 0.2]),
        fo.Detection(label="person", bounding_box=[0.0, 0.0, 0.1, 0.1]),
    ])
    merged = cvat_load.merge_class_detections(before, after, names)
    by_label = {detection.label: detection.bounding_box for detection in merged.detections}
    assert set(by_label) == {"adult_head", "baby_head"}
    assert by_label["adult_head"] == [0.1, 0.1, 0.2, 0.2]
    assert by_label["baby_head"] == [0.6, 0.6, 0.2, 0.2]
    assert cvat_load.boxes_changed(before, merged, names)
    jitter = fo.Detections(detections=[
        fo.Detection(label="baby_head", bounding_box=[0.2001, 0.2, 0.2, 0.2]),
    ])
    original = fo.Detections(detections=[
        fo.Detection(label="baby_head", bounding_box=[0.2, 0.2, 0.2, 0.2]),
    ])
    assert not cvat_load.boxes_changed(original, jitter, names)
    assert cvat_load.append_changed_tags(["relabel", "changed"], ["label_cvat_261012"]) == [
        "relabel",
        "changed",
        "label_cvat_261012",
    ]


def test_apply_load_tags_changed_samples(tmp_path):
    dataset = fo.Dataset()
    try:
        path = tmp_path / "a.jpg"
        Image.new("RGB", (12, 12)).save(path)
        sample = fo.Sample(
            filepath=str(path),
            tags=["relabel", "cvat_cvat_head_261012"],
            ground_truth=fo.Detections(detections=[
                fo.Detection(label="adult_head", bounding_box=[0.1, 0.1, 0.2, 0.2]),
                fo.Detection(label="baby_head", bounding_box=[0.3, 0.3, 0.2, 0.2]),
            ]),
        )
        sample_id = str(dataset.add_sample(sample))
        previous = dataset[sample_id].ground_truth
        dataset[sample_id].ground_truth = fo.Detections(detections=[
            fo.Detection(label="adult_head", bounding_box=[0.8, 0.8, 0.1, 0.1]),
            fo.Detection(label="baby_head", bounding_box=[0.4, 0.4, 0.2, 0.2]),
        ])
        dataset[sample_id].save()
        changed, unchanged = cvat_load.apply_load(
            dataset,
            [sample_id],
            {sample_id: previous},
            "ground_truth",
            {"baby_head"},
            ["label_cvat_261012"],
        )
        assert (changed, unchanged) == (1, 0)
        loaded = dataset[sample_id]
        boxes = {detection.label: detection.bounding_box for detection in loaded.ground_truth.detections}
        assert boxes["adult_head"] == [0.1, 0.1, 0.2, 0.2]
        assert boxes["baby_head"] == [0.4, 0.4, 0.2, 0.2]
        assert loaded.tags == ["relabel", "cvat_cvat_head_261012", "label_cvat_261012", "changed"]
    finally:
        dataset.delete()


def test_label_field_comes_from_the_run():
    info = SimpleNamespace(config=SimpleNamespace(label_schema={"ground_truth": {}}))
    assert cvat_load.label_field_from_run(info) == "ground_truth"
    with pytest.raises(ValueError, match="exactly one"):
        cvat_load.label_field_from_run(SimpleNamespace(config=SimpleNamespace(label_schema={})))


def test_load_cli_requires_writeback_tags():
    args = cvat_load.parse_args(
        [
            "--dataset", "BBM08S_head",
            "--anno-key", "cvat_head_261012",
            "--classes", "baby_head",
            "--sample-tags", "label_cvat_261012",
            "--dry-run",
        ]
    )
    assert args.classes == ["baby_head"]
    assert args.sample_tags == ["label_cvat_261012"]
    assert args.dry_run
