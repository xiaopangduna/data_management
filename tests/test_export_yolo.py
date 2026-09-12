"""Check generated names and real YOLO media/annotation pairing."""
import importlib.util
from pathlib import Path

import fiftyone as fo
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "export_yolo", Path(__file__).parents[1] / "scripts/export_yolo.py"
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def labels(*names):
    return fo.Detections(detections=[
        fo.Detection(label=name, bounding_box=[0, 0, .5, .5]) for name in names
    ])


def test_cli_excludes_dup_repeat_drop_by_default():
    args = export.parse_args([
        "--dataset", "demo", "--output-dir", "out", "--tags", "train",
    ])
    assert args.exclude_tags == ["dup_repeat_drop"]
    assert args.split == "train"
    args = export.parse_args([
        "--dataset", "demo", "--output-dir", "out", "--tags", "train",
        "--exclude-tags", "dup_repeat_drop,dup_near",
        "--split", "train_v003",
    ])
    assert args.exclude_tags == ["dup_repeat_drop", "dup_near"]
    assert args.split == "train_v003"


def test_check_output_allows_other_splits(tmp_path):
    output = tmp_path / "out"
    train_images = output / "images" / "train"
    train_images.mkdir(parents=True)
    (train_images / "a.jpg").write_bytes(b"x")
    (output / "dataset.yaml").write_text("names: {0: baby_head}\n")
    (output / "images" / "test").mkdir(parents=True)
    export.check_output(output, "test")


def test_check_output_rejects_same_split(tmp_path):
    output = tmp_path / "out"
    test_images = output / "images" / "test"
    test_images.mkdir(parents=True)
    (test_images / "a.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="empty or absent"):
        export.check_output(output, "test")


def test_name_rules():
    classes = ["baby_head", "adult_head"]
    assert export.export_stem(labels("adult_head", "baby_head", "baby_head"), classes, 1) == "baby_head__adult_head_000001"
    assert export.export_stem(labels(), classes, 2) == "negative_000002"
    assert export.export_stem(None, classes, 3) == "negative_000003"
    assert export.export_stem(labels("adult_head"), classes, 1000000) == "adult_head_1000000"
    many = [f"c{i}" for i in range(7)]
    assert export.export_stem(labels(*many), many, 4) == "c0__c1__c2__c3__c4__c5__more_000004"
    unsafe = ["../baby/head", "头" * 100]
    name = export.export_stem(labels(*unsafe), unsafe, 5)
    assert "/" not in name and len(name.encode()) < 240


@pytest.mark.parametrize("mode", ["copy", "symlink"])
def test_export_duplicate_names_and_negatives(tmp_path, mode):
    sources = []
    for folder, color in [("a", "red"), ("b", "blue")]:
        source = tmp_path / folder / "same.jpg"
        source.parent.mkdir()
        Image.new("RGB", (10, 10), color).save(source)
        sources.append(source)
    output = tmp_path / "out"
    with export.make_exporter(output, ["baby_head", "adult_head"], mode) as writer:
        writer.export_sample(str(sources[0]), labels("adult_head", "baby_head"))
        writer.export_sample(str(sources[1]), labels("baby_head"))
        writer.export_sample(str(sources[0]), None)
        writer.export_sample(str(sources[0]), labels())
    stems = ["baby_head__adult_head_000001", "baby_head_000002", "negative_000003", "negative_000004"]
    for stem, source in zip(stems, [sources[0], sources[1], sources[0], sources[0]]):
        image = output / "images/train" / (stem + ".jpg")
        annotation = output / "labels/train" / (stem + ".txt")
        assert image.read_bytes() == source.read_bytes()
        assert image.is_symlink() == (mode == "symlink")
        assert annotation.is_file()
    assert (output / "labels/train/negative_000003.txt").read_text() == ""
    assert (output / "labels/train/negative_000004.txt").read_text() == ""
    lines = (output / "labels/train/baby_head__adult_head_000001.txt").read_text().splitlines()
    assert [line.split()[0] for line in lines] == ["1", "0"]
    assert (output / "dataset.yaml").is_file()


def test_export_uses_custom_split_folder(tmp_path):
    source = tmp_path / "src.jpg"
    Image.new("RGB", (10, 10), "red").save(source)
    output = tmp_path / "out"
    with export.make_exporter(output, ["baby_head"], "copy", split="train_v003") as writer:
        writer.export_sample(str(source), labels("baby_head"))
    image = output / "images/train_v003/baby_head_000001.jpg"
    annotation = output / "labels/train_v003/baby_head_000001.txt"
    assert image.is_file() and not image.is_symlink()
    assert annotation.is_file()
    assert not (output / "images/train").exists()
