"""Exercise conversion using real images and YOLO sidecars, without a database."""
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "convert_yolo_to_xlabel", Path(__file__).parents[1] / "scripts/convert_yolo_to_xlabel.py")
convert = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(convert)


@pytest.fixture
def task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    images, labels = tmp_path / "images", tmp_path / "labels"
    (images / "train").mkdir(parents=True)
    (labels / "train").mkdir(parents=True)
    Image.new("RGB", (100, 50), "red").save(images / "train/a.jpg")
    (labels / "train/a.txt").write_text("1 0.5 0.5 0.4 0.4\n")
    return ["--images-dir", str(images), "--labels-dir", str(labels),
            "--out-dir", str(tmp_path / "output"), "--class-names", "person,baby_head"]


@pytest.mark.parametrize("mode", ["none", "symlink", "copy"])
def test_media_and_geometry(task, tmp_path, mode):
    assert convert.main(task + ["--export-media", mode]) == 0
    document = json.loads((tmp_path / "output/train/a.json").read_text())
    assert document["imageWidth"] == 100 and document["imageHeight"] == 50
    assert document["shapes"][0]["label"] == "baby_head"
    assert document["shapes"][0]["points"] == [[30, 15], [70, 35]]
    assert document["checked"] is False and "sample_id" not in document
    media = tmp_path / "output/train/a.jpg"
    original = tmp_path / "images/train/a.jpg"
    if mode == "none":
        assert document["imagePath"] == str(original)
        assert not media.exists()
    else:
        assert document["imagePath"] == "a.jpg"
        assert media.read_bytes() == original.read_bytes()
        assert media.is_symlink() == (mode == "symlink")


def test_empty_missing_invalid_and_collision(task, tmp_path):
    images, labels = tmp_path / "images/train", tmp_path / "labels/train"
    (labels / "a.txt").write_text("")
    for name in ("missing", "invalid", "collision"):
        Image.new("RGB", (10, 10)).save(images / f"{name}.jpg")
    (labels / "invalid.txt").write_text("0 .5 .5 .2 .2\n9 .5 .5 .2 .2")
    (labels / "collision.txt").write_text("")
    Image.new("RGB", (10, 10)).save(images / "collision.png")
    assert convert.main(task) == 1
    output = tmp_path / "output/train"
    assert json.loads((output / "a.json").read_text())["shapes"] == []
    assert sorted(p.name for p in output.iterdir()) == ["a.json"]
    report = (tmp_path / "tmp/convert_yolo_to_xlabel_issues.csv").read_text()
    assert all(kind in report for kind in ("missing_label", "invalid_input", "path_collision"))


@pytest.mark.parametrize("row", ["0 nan .5 .1 .1", "0 .5 .5 0 .1", "0 .9 .5 .4 .2",
                                 "0 .5 .5 .2 .2 .9", "0.0 .5 .5 .2 .2"])
def test_reject_invalid_rows(task, tmp_path, row):
    (tmp_path / "labels/train/a.txt").write_text(row)
    assert convert.main(task) == 1
    assert not (tmp_path / "output").exists()


def test_dry_run_and_overwrite_preserve_sources(task, tmp_path):
    assert convert.main(task + ["--dry-run", "--export-media", "copy"]) == 0
    assert not (tmp_path / "output").exists()
    assert (tmp_path / "tmp/convert_yolo_to_xlabel_issues.csv").exists()
    assert convert.main(task + ["--export-media", "symlink"]) == 0
    original = tmp_path / "images/train/a.jpg"
    source_bytes = original.read_bytes()
    json_path = tmp_path / "output/train/a.json"
    before = json_path.read_bytes()
    (tmp_path / "labels/train/a.txt").write_text("")
    assert convert.main(task) == 1
    assert json_path.read_bytes() == before
    assert convert.main(task + ["--export-media", "copy", "--overwrite"]) == 0
    assert not (tmp_path / "output/train/a.jpg").is_symlink()
    assert original.read_bytes() == source_bytes
    assert json.loads(json_path.read_text())["shapes"] == []


def test_class_mapping_rejects_empty_slots(task):
    with pytest.raises(SystemExit):
        convert.parse_args(task[:-1] + ["person,,baby_head"])


def test_output_tree_must_be_separate(task, tmp_path):
    args = convert.parse_args(task)
    args.out_dir = tmp_path / "images/export"
    with pytest.raises(ValueError, match="separate"):
        convert.convert(args)
