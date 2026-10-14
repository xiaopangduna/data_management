"""Copy label tags onto fields, then write those fields back to tags."""
import importlib.util
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


attributes = load_script("update_label_attributes")
tags_from = load_script("update_tags_from_attributes")
AGE = attributes.FieldSpec("age", 0, 2)


def test_attribute_from_tags_uses_the_highest_code_in_range():
    assert attributes.attribute_from_tags(["ignore", "age_0", "age_2"], AGE) == ("2", [])
    assert attributes.attribute_from_tags(["ignore"], AGE) == ("0", [])
    assert attributes.attribute_from_tags(["age_1"], AGE) == ("1", [])
    assert attributes.attribute_from_tags(["age_3", "age_2"], AGE) == (None, ["age_3"])
    assert attributes.attribute_from_tags(None, AGE) == ("0", [])


def test_field_spec_rejects_a_reversed_range():
    args = attributes.parse_args(
        [
            "--dataset", "BBM08S_head",
            "--sample-tags", "pred_v121_bad",
            "--classes", "head",
            "--fields", "age:0-2",
            "--fields", "occ:0-1",
            "--dry-run",
        ]
    )
    assert [(spec.name, spec.low, spec.high) for spec in args.fields] == [("age", 0, 2), ("occ", 0, 1)]
    with pytest.raises(SystemExit):
        attributes.parse_args(
            ["--dataset", "BBM08S_head", "--sample-tags", "relabel", "--classes", "head", "--fields", "age"]
        )
    with pytest.raises(SystemExit):
        attributes.parse_args(
            ["--dataset", "BBM08S_head", "--sample-tags", "relabel", "--classes", "head", "--fields", "age:2-0"]
        )


def test_apply_attributes_writes_fields_and_keeps_tags(tmp_path):
    dataset = fo.Dataset()
    try:
        path = tmp_path / "a.jpg"
        Image.new("RGB", (8, 8)).save(path)
        sample_id = str(dataset.add_sample(fo.Sample(
            filepath=str(path),
            tags=["pred_v121_bad"],
            ground_truth=fo.Detections(detections=[
                fo.Detection(label="head", tags=["age_0", "age_2", "ignore"], bounding_box=[0.1, 0.1, 0.2, 0.2]),
                fo.Detection(label="head", tags=["age_9"], bounding_box=[0.3, 0.3, 0.2, 0.2]),
                fo.Detection(label="person", tags=["age_1"], bounding_box=[0.5, 0.5, 0.2, 0.2]),
            ]),
        )))
        updated, unchanged, issues = attributes.apply_attributes(
            dataset, [sample_id], "ground_truth", {"head"}, [AGE], dry_run=False,
        )
        assert (updated, unchanged) == (1, 1)
        assert issues[0]["issue"] == "out_of_range"
        assert issues[0]["detail"] == "age_9"
        loaded = dataset[sample_id].ground_truth.detections
        assert loaded[0].get_attribute_value("age") == "2"
        assert loaded[0].tags == ["age_0", "age_2", "ignore"]
        assert loaded[1].has_attribute("age") is False
        assert loaded[2].has_attribute("age") is False
        assert dataset[sample_id].tags == ["pred_v121_bad"]
    finally:
        dataset.delete()


def test_dry_run_does_not_write_fields(tmp_path):
    dataset = fo.Dataset()
    try:
        path = tmp_path / "a.jpg"
        Image.new("RGB", (8, 8)).save(path)
        sample_id = str(dataset.add_sample(fo.Sample(
            filepath=str(path),
            tags=["relabel"],
            ground_truth=fo.Detections(detections=[
                fo.Detection(label="head", tags=["age_1"], bounding_box=[0.1, 0.1, 0.2, 0.2]),
            ]),
        )))
        updated, unchanged, issues = attributes.apply_attributes(
            dataset, [sample_id], "ground_truth", {"head"}, [AGE], dry_run=True,
        )
        assert (updated, unchanged, issues) == (1, 0, [])
        assert dataset[sample_id].ground_truth.detections[0].has_attribute("age") is False
    finally:
        dataset.delete()


def test_tags_for_value_replace_only_that_family():
    spec = tags_from.FieldSpec("age", 0, 2)
    assert tags_from.tags_for_value(["ignore", "age_0", "age_2"], spec, "1") == (["ignore", "age_1"], True)
    assert tags_from.tags_for_value(["age_2"], spec, "2") == (["age_2"], False)
    assert tags_from.tags_for_value(["ignore"], spec, "0") == (["ignore", "age_0"], True)
    assert tags_from.tags_for_value(None, spec, "0") == (["age_0"], True)


def test_apply_tag_sync_marks_only_disagreements(tmp_path):
    dataset = fo.Dataset()
    try:
        path = tmp_path / "a.jpg"
        Image.new("RGB", (8, 8)).save(path)
        same = fo.Detection(label="head", tags=["age_2", "ignore"], bounding_box=[0.1, 0.1, 0.2, 0.2])
        same.set_attribute_value("age", "2")
        edited = fo.Detection(label="head", tags=["age_0"], bounding_box=[0.3, 0.3, 0.2, 0.2])
        edited.set_attribute_value("age", "1")
        created = fo.Detection(label="head", tags=[], bounding_box=[0.4, 0.4, 0.2, 0.2])
        created.set_attribute_value("age", "0")
        missing = fo.Detection(label="head", tags=["age_2"], bounding_box=[0.5, 0.5, 0.2, 0.2])
        other = fo.Detection(label="person", tags=["age_0"], bounding_box=[0.6, 0.6, 0.2, 0.2])
        other.set_attribute_value("age", "1")
        changed_id = str(dataset.add_sample(fo.Sample(
            filepath=str(path),
            tags=["cvat_cvat_head_261013"],
            ground_truth=fo.Detections(detections=[same, edited, created, missing, other]),
        )))
        still_path = tmp_path / "b.jpg"
        Image.new("RGB", (8, 8)).save(still_path)
        still = fo.Detection(label="head", tags=["age_0"], bounding_box=[0.1, 0.1, 0.2, 0.2])
        still.set_attribute_value("age", "0")
        still_id = str(dataset.add_sample(fo.Sample(
            filepath=str(still_path),
            tags=["cvat_cvat_head_261013"],
            ground_truth=fo.Detections(detections=[still]),
        )))
        changed, unchanged, issues = tags_from.apply_tag_sync(
            dataset,
            [changed_id, still_id],
            "ground_truth",
            {"head"},
            [tags_from.FieldSpec("age", 0, 2)],
            dry_run=False,
        )
        assert (changed, unchanged) == (1, 1)
        assert [row["issue"] for row in issues] == ["missing_attribute"]
        loaded = dataset[changed_id]
        boxes = loaded.ground_truth.detections
        assert boxes[0].tags == ["age_2", "ignore"]
        assert boxes[1].tags == ["age_1"]
        assert boxes[2].tags == ["age_0"]
        assert boxes[3].tags == ["age_2"]
        assert boxes[4].tags == ["age_0"]
        assert loaded.tags == ["cvat_cvat_head_261013", "changed"]
        assert dataset[still_id].tags == ["cvat_cvat_head_261013"]
    finally:
        dataset.delete()
