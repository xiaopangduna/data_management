"""Rename rules for baby_head and adult_head."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "update_head_labels", Path(__file__).parents[1] / "scripts/update_head_labels.py"
)
update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update)


def test_renamed_tags_keep_existing_age_and_other_tags():
    assert update.renamed_tags("baby_head", ["face_invisibility_1", "age_0"]) == (
        "head", ["face_invisibility_1", "age_0"],
    )
    assert update.renamed_tags("adult_head", ["review", "age_2"]) == (
        "head", ["review", "age_2"],
    )
    assert update.renamed_tags("baby_head", None) == ("head", ["age_0"])
    assert update.renamed_tags("baby_head", ["age_1", "age_0"]) == ("head", ["age_1", "age_0"])
    assert update.renamed_tags("baby_body", ["age_0"]) is None
    assert update.renamed_tags("head", ["age_0"]) is None


def test_extra_age_tags_ignore_the_expected_value():
    assert update.extra_age_tags(["age_1", "age_0", "review"], "age_0") == ["age_1"]
    assert update.extra_age_tags(["age_2"], "age_2") == []
    assert update.extra_age_tags(None, "age_0") == []


def test_pipeline_renames_only_source_labels():
    pipeline = update.rename_pipeline("ground_truth", now=None)
    switch = pipeline[0]["$set"]["ground_truth.detections"]["$map"]["in"]["$switch"]
    assert [branch["case"]["$eq"][1] for branch in switch["branches"]] == ["baby_head", "adult_head"]
    assert switch["default"] == "$$det"
    baby = switch["branches"][0]["then"]["$mergeObjects"][1]
    assert baby["label"] == "head"
    assert baby["tags"]["$cond"]["if"] == {"$in": ["age_0", {"$ifNull": ["$$det.tags", []]}]}
