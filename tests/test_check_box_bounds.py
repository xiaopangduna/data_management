"""Box bound and same-label IoU checks."""
import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_box_bounds", Path(__file__).parents[1] / "scripts/check_box_bounds.py"
)
check = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = check
spec.loader.exec_module(check)


def test_inside_and_border_touch_are_accepted():
    assert check.classify_box([0.1, 0.2, 0.3, 0.4]) is None
    assert check.classify_box([0.0, 0.0, 1.0, 1.0]) is None
    assert check.classify_box([0.2, -1e-6, 0.5, 0.4]) is None


def test_boxes_outside_the_tolerance_are_split_by_overlap():
    assert check.classify_box([2.4, 0.1, 0.2, 0.2]) == "no_overlap"
    assert check.classify_box([-0.5, 0.1, 0.2, 0.2]) == "no_overlap"
    assert check.classify_box([-0.2, 0.1, 0.5, 0.2]) == "partial"
    assert check.classify_box([0.9, 0.1, 0.2, 0.2]) == "partial"
    assert check.classify_box([0.2, -1e-6, 0.5, 0.4], tol=0) == "partial"


def test_invalid_boxes():
    assert check.classify_box(None) == "invalid"
    assert check.classify_box([0.1, 0.2, 0.3]) == "invalid"
    assert check.classify_box([0.1, 0.2, 0.0, 0.3]) == "invalid"
    assert check.classify_box([0.1, math_nan(), 0.2, 0.2]) == "invalid"


def test_overlap_and_iou():
    assert check.overlap_fraction(-0.2, 0.1, 0.5, 0.2) == pytest.approx(0.6)
    assert check.max_overflow(-0.2, 0.1, 0.5, 0.2) == pytest.approx(0.2)
    assert check.box_iou((0, 0, 0.2, 0.2), (0, 0, 0.2, 0.2)) == 1
    assert check.box_iou((0, 0, 0.2, 0.2), (0.1, 0, 0.2, 0.2)) == pytest.approx(1 / 3)
    assert check.box_iou((0, 0, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)) == 0


def test_rows_keep_only_abnormal_boxes_and_their_index():
    rows = check.rows_for_sample(
        "sample-1",
        "/img.jpg",
        ["train", "review"],
        ["head", "baby_body", "head"],
        [[0.1, 0.1, 0.2, 0.2], [2.0, 0.2, 0.3, 0.3], [-0.4, 0.2, 0.5, 0.2]],
        check.DEFAULT_EDGE_TOL,
    )
    assert [row["issue"] for row in rows] == ["no_overlap", "partial"]
    assert [row["action"] for row in rows] == ["delete", "delete"]
    assert [row["det_index"] for row in rows] == ["1", "2"]
    assert rows[0]["label"] == "baby_body"
    assert rows[0]["tags"] == "train,review"
    assert rows[0]["overlap"] == "0.000000"
    assert rows[1]["x"] == "-0.400000"


def test_slight_overflow_is_clipped_and_tags_the_sample():
    plan = check.plan_sample(
        ["head"],
        [[-0.01, 0.2, 0.3, 0.4]],
        ["train"],
        check.DEFAULT_EDGE_TOL,
    )
    assert plan.edits == (check.BoxEdit(0, "clip", (0.0, 0.2, 0.29, 0.4)),)
    assert plan.tags == ["train", "bad_box", "changed_box"]


def test_width_past_the_image_is_deleted():
    plan = check.plan_sample(
        ["head"],
        [[0.0, 0.1, 1.01, 0.2]],
        ["train"],
        check.DEFAULT_EDGE_TOL,
    )
    assert plan.edits == (check.BoxEdit(0, "delete"),)
    assert plan.tags == ["train", "bad_box", "changed_box"]


def test_high_iou_tags_the_sample_and_keeps_the_boxes():
    plan = check.plan_sample(
        ["head", "head", "baby_body"],
        [[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2]],
        ["train", "changed_box"],
        check.DEFAULT_EDGE_TOL,
        0.9,
    )
    assert plan.edits == ()
    assert plan.tags == ["train", "bad_box"]
    assert [issue[0] for issue in plan.issues] == ["high_iou", "high_iou"]


def test_high_iou_row_records_the_partner():
    rows = check.rows_for_sample(
        "sample-1",
        "/img.jpg",
        [],
        ["head", "head"],
        [[0.1, 0.1, 0.2, 0.2], [0.11, 0.1, 0.2, 0.2]],
        check.DEFAULT_EDGE_TOL,
        0.5,
    )
    assert [row["issue"] for row in rows] == ["high_iou", "high_iou"]
    assert [row["action"] for row in rows] == ["keep", "keep"]
    assert rows[0]["pair_index"] == "1"
    assert rows[1]["pair_index"] == "0"
    assert float(rows[0]["iou"]) >= 0.5


def test_outside_box_with_high_iou_is_deleted():
    plan = check.plan_sample(
        ["head", "head"],
        [[2.0, 0.1, 0.2, 0.2], [2.0, 0.1, 0.2, 0.2]],
        ["train"],
        check.DEFAULT_EDGE_TOL,
        0.9,
    )
    assert [edit.action for edit in plan.edits] == ["delete", "delete"]
    assert plan.tags == ["train", "bad_box", "changed_box"]
    assert {issue[0] for issue in plan.issues} == {"no_overlap", "high_iou"}


def test_stale_sample_tags_are_removed_when_boxes_are_clean():
    stale = check.plan_sample(
        ["head"],
        [[0.1, 0.1, 0.2, 0.2]],
        ["train", "bad_box", "changed_box"],
        check.DEFAULT_EDGE_TOL,
        0.9,
    )
    assert stale.edits == ()
    assert stale.tags == ["train"]
    clean = check.plan_sample(
        ["head"],
        [[0.1, 0.1, 0.2, 0.2]],
        ["train"],
        check.DEFAULT_EDGE_TOL,
        0.9,
    )
    assert clean.tags is None
    assert clean.edits == ()


def test_iou_below_threshold_is_ignored():
    plan = check.plan_sample(
        ["head", "head"],
        [[0, 0, 0.2, 0.2], [0.1, 0, 0.2, 0.2]],
        [],
        check.DEFAULT_EDGE_TOL,
        0.9,
    )
    assert plan.issues == ()
    assert plan.edits == ()
    assert plan.tags is None


def test_dry_run_is_opt_in():
    assert check.parse_args(["--dataset", "BBM08S_head"]).dry_run is False
    assert check.parse_args(["--dataset", "BBM08S_head", "--dry-run"]).dry_run is True


def test_thresholds_reject_values_that_swallow_the_check():
    with pytest.raises(Exception):
        check.edge_tol("0.5")
    with pytest.raises(Exception):
        check.edge_tol("-0.1")
    with pytest.raises(Exception):
        check.iou_threshold("0")
    with pytest.raises(Exception):
        check.iou_threshold("1.1")
    with pytest.raises(Exception):
        check.clip_overflow("0")
    with pytest.raises(Exception):
        check.clip_overflow("0.5")
    assert check.edge_tol("0") == 0.0
    assert check.iou_threshold("1") == 1.0
    assert check.clip_overflow("0.02") == 0.02


def math_nan() -> float:
    return float("nan")
