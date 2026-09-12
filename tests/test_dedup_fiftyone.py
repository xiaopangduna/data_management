"""Exact/near grouping, keep/drop invariants, and re-run behavior."""
import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "dedup_fiftyone", Path(__file__).parents[1] / "scripts/dedup_fiftyone.py"
)
dedup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dedup
spec.loader.exec_module(dedup)


def make_ref(
    sample_id: str,
    *,
    filepath: str = "",
    relpath: str = "",
    sha256: str = "",
    phash: str = "",
    area: int = 100,
    tags: tuple[str, ...] = (),
    dup_group: str = "",
    dup_of: str = "",
) -> dedup.SampleRef:
    path = filepath or f"/images/{sample_id}.jpg"
    return dedup.SampleRef(
        id=sample_id,
        filepath=path,
        relpath=relpath or Path(path).name,
        sha256=sha256,
        phash=phash,
        area=area,
        tags=tags,
        dup_group=dup_group,
        dup_of=dup_of,
    )


def planned(ref: dedup.SampleRef, plan: dedup.DedupPlan) -> dedup.SampleUpdate:
    return plan.updates.get(
        ref.id,
        dedup.SampleUpdate(tags=ref.tags, dup_group=ref.dup_group, dup_of=ref.dup_of),
    )


def test_cli_tag_only_by_default():
    args = dedup.parse_args(["--dataset-name", "demo"])
    assert args.dataset_name == "demo"
    assert args.hamming_max == 2
    assert not args.dry_run
    dry = dedup.parse_args(["--dataset-name", "demo", "--dry-run"])
    assert dry.dry_run
    with pytest.raises(SystemExit):
        dedup.parse_args(["--dataset-name", "demo", "--apply-deletes"])


def test_new_exact_group_keep_drop_and_fields():
    keep = make_ref("a", relpath="a.jpg", filepath="/data/a.jpg", sha256="abc")
    drop = make_ref("b", relpath="b.jpg", filepath="/data/b.jpg", sha256="abc")
    unique = make_ref("c", relpath="c.jpg", sha256="def")
    plan = dedup.build_dedup_plan([keep, drop, unique], hamming_max=0)
    assert list(plan.exact_groups) == ["abc"]
    assert plan.exact_keep == 1 and plan.exact_drop == 1 and plan.exact_keep_conflict == 0
    assert plan.drop_ids == ["b"]
    keep_update = planned(keep, plan)
    drop_update = planned(drop, plan)
    assert keep_update.tags == ("dup_repeat", "dup_repeat_keep")
    assert drop_update.tags == ("dup_repeat", "dup_repeat_drop")
    assert keep_update.dup_group == drop_update.dup_group == "exact:abc"
    assert keep_update.dup_of == ""
    assert drop_update.dup_of == "/data/a.jpg"
    assert unique.id not in plan.updates


def test_near_includes_all_members_and_excludes_exact_drops():
    keep = make_ref("a", relpath="a.jpg", filepath="/data/a.jpg", sha256="aaa", phash="00")
    drop = make_ref("b", relpath="b.jpg", filepath="/data/b.jpg", sha256="aaa", phash="00")
    near = make_ref("c", relpath="c.jpg", filepath="/data/c.jpg", sha256="ccc", phash="01")
    plan = dedup.build_dedup_plan([keep, drop, near], hamming_max=2)
    assert plan.drop_ids == ["b"]
    near_ids = {ref.id for group in plan.near_groups.values() for ref in group}
    assert near_ids == {"a", "c"}
    assert "b" not in near_ids
    keep_update = planned(keep, plan)
    near_update = planned(near, plan)
    drop_update = planned(drop, plan)
    assert "dup_near" in keep_update.tags and "dup_repeat_keep" in keep_update.tags
    assert keep_update.tags[:2] == ("dup_repeat", "dup_repeat_keep")
    assert near_update.tags == ("dup_near",)
    assert "dup_near" not in drop_update.tags
    assert keep_update.dup_group == "exact:aaa"
    assert near_update.dup_group.startswith("near:")


def test_re_run_preserves_human_keep_drop_swap():
    keep = make_ref(
        "a",
        relpath="a.jpg",
        filepath="/data/a.jpg",
        sha256="abc",
        tags=("dup_repeat", "dup_repeat_drop"),
    )
    drop = make_ref(
        "b",
        relpath="b.jpg",
        filepath="/data/b.jpg",
        sha256="abc",
        tags=("dup_repeat", "dup_repeat_keep"),
    )
    plan = dedup.build_dedup_plan([keep, drop], hamming_max=0)
    decision = plan.exact_decisions["abc"]
    assert decision.roles == {"a": "drop", "b": "keep"}
    assert decision.keeper.id == "b"
    assert plan.drop_ids == ["a"]
    assert planned(keep, plan).tags == ("dup_repeat", "dup_repeat_drop")
    assert planned(drop, plan).tags == ("dup_repeat", "dup_repeat_keep")
    assert planned(keep, plan).dup_of == "/data/b.jpg"


def test_new_member_of_decided_group_becomes_drop():
    keep = make_ref(
        "a",
        relpath="a.jpg",
        filepath="/data/a.jpg",
        sha256="abc",
        tags=("dup_repeat", "dup_repeat_keep"),
    )
    extra = make_ref("c", relpath="c.jpg", filepath="/data/c.jpg", sha256="abc")
    plan = dedup.build_dedup_plan([keep, extra], hamming_max=0)
    assert plan.exact_decisions["abc"].roles == {"a": "keep", "c": "drop"}
    assert planned(extra, plan).tags == ("dup_repeat", "dup_repeat_drop")


def test_zero_or_multiple_keeps_are_not_rewritten(caplog):
    first = make_ref(
        "a",
        relpath="a.jpg",
        sha256="abc",
        tags=("dup_repeat", "dup_repeat_keep"),
    )
    second = make_ref(
        "b",
        relpath="b.jpg",
        sha256="abc",
        tags=("dup_repeat", "dup_repeat_keep"),
    )
    multi = dedup.build_dedup_plan([first, second], hamming_max=0)
    assert multi.exact_keep_conflict == 1
    assert multi.exact_decisions["abc"].roles == {"a": "keep", "b": "keep"}
    assert not multi.drop_ids
    assert "2 keep tags" in caplog.text

    only_drop = make_ref("d", relpath="d.jpg", sha256="def", tags=("dup_repeat", "dup_repeat_drop"))
    other_drop = make_ref("e", relpath="e.jpg", sha256="def", tags=("dup_repeat", "dup_repeat_drop"))
    empty = dedup.build_dedup_plan([only_drop, other_drop], hamming_max=0)
    assert empty.exact_keep_conflict == 1
    assert empty.exact_decisions["def"].keeper is None
    assert empty.exact_decisions["def"].roles == {"d": "drop", "e": "drop"}
    assert "no keep" in caplog.text


def test_stale_dup_tags_cleared_when_group_dissolves():
    leftover = make_ref(
        "a",
        relpath="a.jpg",
        sha256="solo",
        tags=("train", "dup_repeat", "dup_repeat_keep", "dup_near"),
        dup_group="exact:old",
        dup_of="/old.jpg",
    )
    plan = dedup.build_dedup_plan([leftover], hamming_max=0)
    update = planned(leftover, plan)
    assert update.tags == ("train",)
    assert update.dup_group == "" and update.dup_of == ""
    assert plan.stale_cleared == 1


def test_report_uses_drop_action_and_prefixed_groups():
    keep = make_ref("a", relpath="a.jpg", sha256="abc", phash="00")
    drop = make_ref("b", relpath="b.jpg", sha256="abc", phash="00")
    near = make_ref("c", relpath="c.jpg", sha256="ccc", phash="01")
    plan = dedup.build_dedup_plan([keep, drop, near], hamming_max=2)
    rows = dedup.build_report_rows("demo", plan)
    exact_rows = [row for row in rows if row["kind"] == "exact"]
    near_rows = [row for row in rows if row["kind"] == "near"]
    assert {row["action"] for row in exact_rows} == {"keep", "drop"}
    assert all(row["dup_group"] == "exact:abc" for row in exact_rows)
    assert "delete" not in {row["action"] for row in rows}
    assert {row["action"] for row in near_rows} == {"keep", "tag_dup_near"}
    assert all(row["dup_group"].startswith("near:") for row in near_rows)


def test_compose_tags_preserves_workflow_tags():
    tags = dedup.compose_tags(("train", "head", "dup_near"), {"dup_repeat", "dup_repeat_keep"})
    assert tags == ("train", "head", "dup_repeat", "dup_repeat_keep")


def test_cluster_identical_phashes():
    assert dedup.cluster_phashes(["aa", "bb", "aa"], 0) == {"aa": "aa", "bb": "bb"}
