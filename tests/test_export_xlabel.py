"""Intersection selection and custom-field X-AnyLabeling export."""
import importlib.util
import sys
from pathlib import Path

import fiftyone as fo
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "export_xlabel", Path(__file__).parents[1] / "scripts/export_xlabel.py"
)
export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export
spec.loader.exec_module(export)


def test_cli_defaults_and_removed_options():
    base = ["--dataset-name", "test", "--out-dir", "out", "--sample-tags", "test,review"]
    args = export.parse_args(base)
    assert args.sample_tags == ["test", "review"]
    assert args.label_field == "ground_truth"
    assert args.labels is None and args.export_labels is None
    for option in ("--include-tags", "--exclude-tags", "--exclude-sample-tags", "--class-names"):
        with pytest.raises(SystemExit):
            export.parse_args(base + [option, "person"])


def test_intersection_and_export_from_same_field(tmp_path):
    dataset = fo.Dataset()
    try:
        image = tmp_path / "image.jpg"
        Image.new("RGB", (100, 100)).save(image)
        def sample(tags, labels):
            return fo.Sample(
                filepath=str(image), tags=tags,
                ground_truth=fo.Detections(detections=[
                    fo.Detection(label="wrong_field", bounding_box=[0, 0, .5, .5])
                ]),
                custom=fo.Detections(detections=[
                    fo.Detection(label=label, bounding_box=[0, 0, .5, .5])
                    for label in labels
                ]) if labels is not None else None,
            )
        samples = [
            sample(["test", "review", "dup_near"], ["person", "adult_head", "baby_head"]),
            sample(["test"], ["person", "adult_head"]),
            sample(["test", "review"], ["person"]),
            sample(["test", "review"], []),
            sample(["test", "review"], None),
        ]
        dataset.add_samples(samples)
        view = export.filtered_view(dataset, ["test", "review"], "custom", ["person", "adult_head"])
        assert view.values("id") == [samples[0].id]
        plan = export.collect_export_plan(view, None, "custom")
        assert [shape["label"] for shape in plan.to_write[0].shapes] == ["person", "adult_head", "baby_head"]
        plan = export.collect_export_plan(view, {"baby_head"}, "custom")
        assert [shape["label"] for shape in plan.to_write[0].shapes] == ["baby_head"]
        assert len(export.filtered_view(dataset, ["test", "review"], "custom")) == 4
        with pytest.raises(ValueError, match="does not exist"):
            export.filtered_view(dataset, ["test"], "missing")
        with pytest.raises(ValueError, match="Detections"):
            export.filtered_view(dataset, ["test"], "filepath")
    finally:
        dataset.delete()
