"""Copy mutually exclusive sample tags onto one sample field."""
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


field = load_script("update_sample_field")
SOURCE = ["aopeng", "boyue"]


def test_parse_args_uses_sample_tags():
    args = field.parse_args(
        ["--dataset", "BBM08S_head", "--field", "vendor", "--sample-tags", "aopeng,boyue", "--dry-run"]
    )
    assert args.field == "vendor"
    assert args.sample_tags == ["aopeng", "boyue"]
    assert args.dry_run is True
    with pytest.raises(SystemExit):
        field.parse_args(["--dataset", "BBM08S_head", "--field", "tags", "--sample-tags", "aopeng"])
    with pytest.raises(SystemExit):
        field.parse_args(["--dataset", "BBM08S_head", "--field", "vendor", "--sample-tags", " "])


def test_plan_sample_writes_exactly_one_tag():
    assert field.plan_sample(["aopeng", "train"], SOURCE, "vendor", None) == ("aopeng", None)
    assert field.plan_sample(["boyue"], SOURCE, "vendor", "") == ("boyue", None)
    assert field.plan_sample(["train"], SOURCE, "vendor", None) == (None, None)
    assert field.plan_sample(["aopeng"], SOURCE, "vendor", "aopeng") == (None, None)

    multiple = field.plan_sample(["boyue", "aopeng"], SOURCE, "vendor", None)[1]
    assert multiple["issue"] == "multiple_tags"
    assert multiple["detail"] == "aopeng,boyue"

    mismatch = field.plan_sample(["aopeng"], SOURCE, "vendor", "boyue")[1]
    assert mismatch["issue"] == "field_mismatch"
    assert mismatch["current"] == "boyue"
    assert mismatch["detail"] == "aopeng"


def add_image(dataset: fo.Dataset, directory: Path, name: str, tags: list[str], vendor: str | None = None) -> str:
    path = directory / name
    Image.new("RGB", (8, 8)).save(path)
    sample = fo.Sample(filepath=str(path), tags=tags)
    if vendor is not None:
        sample["vendor"] = vendor
    return str(dataset.add_sample(sample))


def test_apply_field_writes_value_and_keeps_tags(tmp_path):
    dataset = fo.Dataset()
    try:
        aopeng_id = add_image(dataset, tmp_path, "a.jpg", ["aopeng", "train"])
        boyue_id = add_image(dataset, tmp_path, "b.jpg", ["boyue"])
        both_id = add_image(dataset, tmp_path, "c.jpg", ["aopeng", "boyue"])
        same_id = add_image(dataset, tmp_path, "d.jpg", ["aopeng"], vendor="aopeng")
        other_id = add_image(dataset, tmp_path, "e.jpg", ["aopeng"], vendor="boyue")
        plain_id = add_image(dataset, tmp_path, "f.jpg", ["train"])
        updated, unchanged, issues = field.apply_field(
            dataset,
            [aopeng_id, boyue_id, both_id, same_id, other_id, plain_id],
            "vendor",
            SOURCE,
            dry_run=False,
        )
        assert (updated, unchanged) == (2, 1)
        assert [row["issue"] for row in issues] == ["multiple_tags", "field_mismatch"]
        assert dataset[aopeng_id]["vendor"] == "aopeng"
        assert dataset[aopeng_id].tags == ["aopeng", "train"]
        assert dataset[boyue_id]["vendor"] == "boyue"
        assert dataset[boyue_id].tags == ["boyue"]
        assert dataset[both_id]["vendor"] is None
        assert dataset[both_id].tags == ["aopeng", "boyue"]
        assert dataset[other_id]["vendor"] == "boyue"
        assert dataset[plain_id]["vendor"] is None
    finally:
        dataset.delete()


def test_dry_run_does_not_write_the_field(tmp_path):
    dataset = fo.Dataset()
    try:
        sample_id = add_image(dataset, tmp_path, "a.jpg", ["aopeng"])
        updated, unchanged, issues = field.apply_field(
            dataset, [sample_id], "vendor", SOURCE, dry_run=True,
        )
        assert (updated, unchanged, issues) == (1, 0, [])
        assert dataset.has_field("vendor") is False
        assert dataset[sample_id].tags == ["aopeng"]
    finally:
        dataset.delete()
