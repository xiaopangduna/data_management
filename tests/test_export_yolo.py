"""Check generated names and real YOLO media/annotation pairing."""
import importlib.util
import sys
from pathlib import Path

import fiftyone as fo
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "export_yolo", Path(__file__).parents[1] / "scripts/export_yolo.py"
)
export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export
spec.loader.exec_module(export)


def labels(*names):
    return fo.Detections(detections=[
        fo.Detection(label=name, bounding_box=[0, 0, .5, .5]) for name in names
    ])


def test_cli_excludes_dup_drops_by_default():
    args = export.parse_args([
        "--dataset", "demo", "--out-dir", "out", "--sample-tags", "train",
    ])
    assert args.exclude_tags == ["dup_repeat_drop", "dup_near_drop"]
    assert args.split == "train"
    args = export.parse_args([
        "--dataset", "demo", "--out-dir", "out", "--sample-tags", "train",
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


def age_spec(*classes):
    return export.parse_attribute_classes(list(classes))


def detection(label, *tags, box=(0, 0, 0.5, 0.5)):
    return fo.Detection(label=label, bounding_box=list(box), tags=list(tags))


def test_attribute_classes_reject_mixed_axes():
    assert export.parse_attribute_classes(["baby_head", "adult_head"]) is None
    spec = age_spec("head-age_0", "head-age_2")
    assert spec.attribute == "age" and spec.default_value == "age_0"
    assert spec.by_label_code[("head", 2)] == "head-age_2"
    mixed = age_spec("head-age_0", "baby_body")
    assert mixed.plain_labels == frozenset({"baby_body"})
    assert mixed.labels == frozenset({"head"})
    with pytest.raises(ValueError, match="overlaps an attribute label"):
        age_spec("head", "head-age_0")
    with pytest.raises(ValueError, match="single attribute"):
        age_spec("head-age_0", "head-eye_0")
    with pytest.raises(ValueError, match="duplicate attribute code"):
        age_spec("head-age_0", "head-age_00")


def test_attribute_resolution_uses_max_tag_or_default():
    spec = age_spec("head-age_0", "head-age_2", "head-age_10")
    bare = detection("head")
    highest = detection("head", "age_0", "age_1", "age_2", "eye_0")
    explicit_zero = detection("head", "age_0")
    other = detection("face", "age_2")
    source = fo.Detections(detections=[bare, highest, explicit_zero, other])
    label, defaulted, omitted, unmapped = export.apply_attribute_labels(source, spec)
    assert [item.label for item in label.detections] == ["head-age_0", "head-age_2", "head-age_0"]
    assert defaulted == 1 and omitted == 0 and unmapped == []
    assert bare.label == "head" and bare.tags == []
    assert highest.tags == ["age_0", "age_1", "age_2", "eye_0"]

    wider = age_spec("head-age_2", "head-age_10")
    label, defaulted, omitted, unmapped = export.apply_attribute_labels(
        fo.Detections(detections=[detection("head", "age_2", "age_10")]), wider,
    )
    assert [item.label for item in label.detections] == ["head-age_10"]
    assert defaulted == 0

    label, defaulted, omitted, unmapped = export.apply_attribute_labels(
        fo.Detections(detections=[detection("head", "age_1"), bare]), spec,
    )
    assert unmapped == [(0, "head-age_1")]
    assert [item.label for item in label.detections] == ["head-age_0"]
    assert defaulted == 1

    only_adult = age_spec("head-age_2")
    label, defaulted, omitted, unmapped = export.apply_attribute_labels(
        fo.Detections(detections=[bare]), only_adult,
    )
    assert label.detections == [] and omitted == 1 and defaulted == 0 and unmapped == []

    eyes = export.parse_attribute_classes(["head-eye_0", "head-eye_1"])
    label, defaulted, omitted, unmapped = export.apply_attribute_labels(
        fo.Detections(detections=[detection("head", "eye_0", "eye_1"), detection("head", "age_2")]),
        eyes,
    )
    assert [item.label for item in label.detections] == ["head-eye_1", "head-eye_0"]
    assert defaulted == 1 and unmapped == []

    mixed = age_spec("head-age_0", "baby_body")
    body = detection("baby_body", "age_2")
    label, defaulted, omitted, unmapped = export.apply_attribute_labels(
        fo.Detections(detections=[bare, body, detection("face")]), mixed,
    )
    assert [item.label for item in label.detections] == ["head-age_0", "baby_body"]
    assert defaulted == 1 and omitted == 0 and unmapped == []
    assert body.label == "baby_body" and body.tags == ["age_2"]


def test_attribute_export_writes_resolved_class_ids(tmp_path):
    source = tmp_path / "src.jpg"
    Image.new("RGB", (10, 10), "red").save(source)
    output = tmp_path / "out"
    classes = ["head-age_0", "head-age_2"]
    spec = export.parse_attribute_classes(classes)
    boxes = fo.Detections(detections=[
        detection("head", "age_2", box=(0.1, 0.2, 0.3, 0.4)),
        detection("head", box=(0.5, 0.5, 0.2, 0.2)),
    ])
    with export.make_exporter(output, classes, "copy", attribute_export=spec) as writer:
        writer.export_sample(str(source), boxes)
    stem = "head-age_0__head-age_2_000001"
    annotation = (output / "labels/train" / f"{stem}.txt").read_text().splitlines()
    assert [line.split()[0] for line in annotation] == ["1", "0"]
    assert [float(part) for part in annotation[0].split()[1:]] == pytest.approx([0.25, 0.4, 0.3, 0.4])
    assert (output / "images/train" / f"{stem}.jpg").is_file()
    assert boxes.detections[0].label == "head"

    mixed_output = tmp_path / "mixed"
    mixed_classes = ["head-age_0", "baby_body"]
    mixed = export.parse_attribute_classes(mixed_classes)
    mixed_boxes = fo.Detections(detections=[
        detection("head", box=(0.1, 0.1, 0.2, 0.2)),
        detection("baby_body", "age_2", box=(0.4, 0.4, 0.3, 0.3)),
        detection("face"),
    ])
    with export.make_exporter(
        mixed_output, mixed_classes, "copy", attribute_export=mixed,
    ) as writer:
        writer.export_sample(str(source), mixed_boxes)
    mixed_stem = "head-age_0__baby_body_000001"
    mixed_lines = (mixed_output / "labels/train" / f"{mixed_stem}.txt").read_text().splitlines()
    assert [line.split()[0] for line in mixed_lines] == ["0", "1"]
    assert mixed_boxes.detections[1].label == "baby_body"


class FakeSample:
    def __init__(self, sample_id, filepath, label):
        self.id = sample_id
        self.filepath = filepath
        self.ground_truth = label

    def __getitem__(self, key):
        return getattr(self, key)


def test_inspect_samples_omits_classes_outside_the_request(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (10, 10), "red").save(image)
    spec = age_spec("head-age_0")
    original = fo.Detections(detections=[
        detection("head"),
        detection("head", "age_0", "age_2"),
        detection("face", "age_2"),
    ])
    samples = [
        FakeSample("keep", str(image), original),
        FakeSample("older", str(image), fo.Detections(detections=[detection("head", "age_1")])),
    ]
    inspected = export.inspect_samples(samples, "ground_truth", spec)
    assert inspected["boxes"] == 1
    assert inspected["defaulted_boxes"] == 1
    assert inspected["omitted_default_boxes"] == 0
    assert inspected["omitted_boxes"] == 2
    assert inspected["negative_images"] == 1
    assert inspected["detected_classes"] == {"head-age_0"}
    assert [item.label for item in original.detections] == ["head", "head", "face"]


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
