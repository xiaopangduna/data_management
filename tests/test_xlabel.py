"""Helpers for X-AnyLabeling attach comparison and tags."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    module_name = name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


attach = load_script("update_xlabel_labels.py")


def test_boxes_equal_allows_two_pixels():
    names = {"person"}
    box = (("person", (0.1, 0.2, 0.3, 0.4)),)
    within = (("person", (0.12, 0.2, 0.3, 0.4)),)
    beyond = (("person", (0.13, 0.2, 0.3, 0.4)),)
    assert attach.boxes_equal(box, box, names, 100, 50)
    assert attach.boxes_equal(box, within, names, 100, 50)
    assert not attach.boxes_equal(box, beyond, names, 100, 50)
    assert attach.boxes_equal(None, (), names, 100, 50)
    assert not attach.boxes_equal(box, (), names, 100, 50)


def test_merge_tags_only_on_changed():
    assert attach.merge_tags(["val2017"], ["label_person_260909"], False, boxes_changed=True) == [
        "val2017",
        "label_person_260909",
        "changed",
    ]
    assert attach.merge_tags(
        ["val2017", "label_person_260909", "changed"],
        ["label_person_260909"],
        False,
        boxes_changed=False,
    ) == ["val2017"]


class PathDataset:
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def values(self, name):
        raise AssertionError("Full dataset values must not be queried")

    def select(self, ids):
        return PathView([(f"{i:024x}", path) for i, path in enumerate(self.filepaths)
                         if f"{i:024x}" in ids])

    def match(self, query):
        paths = query["filepath"]["$in"]
        assert len(paths) <= attach.QUERY_BATCH_SIZE
        return PathView([(f"{i:024x}", path) for i, path in enumerate(self.filepaths) if path in paths])

    def has_field(self, name):
        return False


class PathView:
    def __init__(self, rows):
        self.rows = rows

    def values(self, fields):
        if fields == "id":
            return [row[0] for row in self.rows]
        assert fields == ["id", "filepath"]
        return [[row[0] for row in self.rows], [row[1] for row in self.rows]]


def write_path_label(label_dir, image_path, sample_id=None):
    import json

    path = label_dir / "train" / "a.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "imagePath": image_path,
        "sample_id": sample_id,
        "imageWidth": 100,
        "imageHeight": 100,
        "shapes": [{"shape_type": "rectangle", "label": "person", "points": [[10, 10], [50, 50]]}],
    }))


def test_attach_paths_relative_absolute_and_symlink(tmp_path):
    labels = tmp_path / "labels"
    images = tmp_path / "images"
    target = images / "train" / "a.jpg"
    target.parent.mkdir(parents=True)
    target.touch()
    link = tmp_path / "linked-images"
    link.symlink_to(images, target_is_directory=True)
    dataset = PathDataset([str(target)])
    for image_path in ("a.jpg", str(target)):
        write_path_label(labels, image_path)
        plan = attach.build_attach_plan(dataset, labels, ["person"], link)
        assert list(plan.to_write) == ["000000000000000000000000"]
        assert not plan.issues


def test_attach_absolute_image_path_falls_back_to_images_dir_basename(tmp_path):
    labels = tmp_path / "labels"
    images = tmp_path / "current-images"
    target = images / "a.jpg"
    target.parent.mkdir()
    target.touch()
    write_path_label(labels, str(tmp_path / "old-images" / "a.jpg"))
    plan = attach.build_attach_plan(PathDataset([str(target)]), labels, ["person"], images)
    assert list(plan.to_write) == ["000000000000000000000000"]
    assert not plan.issues


def test_attach_paths_do_not_guess_basename_or_duplicate_sample(tmp_path):
    labels = tmp_path / "labels"
    images = tmp_path / "images"
    write_path_label(labels, "a.jpg")
    wrong_directory = PathDataset([str(images / "other" / "a.jpg")])
    plan = attach.build_attach_plan(wrong_directory, labels, ["person"], images)
    assert [row["issue"] for row in plan.issues] == ["orphan_image_path"]
    assert not plan.to_write
    target = str(images / "train" / "a.jpg")
    plan = attach.build_attach_plan(PathDataset([target, target]), labels, ["person"], images)
    assert [row["issue"] for row in plan.issues] == ["ambiguous_image_path"]
    assert not plan.to_write


def test_attach_id_priority_and_missing_fields(tmp_path):
    labels = tmp_path / "labels"
    images = tmp_path / "images"
    dataset = PathDataset([str(images / "train" / "a.jpg")])
    write_path_label(labels, "wrong.jpg", "000000000000000000000000")
    plan = attach.build_attach_plan(dataset, labels, ["person"], images)
    assert list(plan.to_write) == ["000000000000000000000000"]
    write_path_label(labels, "a.jpg", "stale-id")
    plan = attach.build_attach_plan(dataset, labels, ["person"], images)
    assert [row["issue"] for row in plan.issues] == ["orphan_label"]
    write_path_label(labels, "")
    plan = attach.build_attach_plan(dataset, labels, ["person"], images)
    assert [row["issue"] for row in plan.issues] == ["missing_image_path"]
    plan = attach.build_attach_plan(dataset, labels, ["person"])
    assert [row["issue"] for row in plan.issues] == ["missing_sample_id"]


def test_converted_media_modes_attach_to_original_image(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.chdir(tmp_path)
    converter = load_script("convert_yolo_to_xlabel.py")
    images, labels = tmp_path / "images", tmp_path / "labels"
    (images / "train").mkdir(parents=True)
    (labels / "train").mkdir(parents=True)
    original = images / "train/a.jpg"
    Image.new("RGB", (100, 50)).save(original)
    (labels / "train/a.txt").write_text("0 .5 .5 .4 .4")
    for mode in ("none", "symlink", "copy"):
        output = tmp_path / mode
        assert converter.main([
            "--images-dir", str(images), "--labels-dir", str(labels),
            "--class-names", "person", "--out-dir", str(output), "--export-media", mode,
        ]) == 0
        plan = attach.build_attach_plan(PathDataset([str(original)]), output, ["person"], images)
        assert not plan.issues
        assert plan.to_write["000000000000000000000000"].boxes == (("person", (0.3, 0.3, 0.4, 0.4)),)


def test_path_queries_are_batched_and_duplicate_json_rejected(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(attach, "QUERY_BATCH_SIZE", 2)
    labels, images = tmp_path / "labels", tmp_path / "images"
    write_path_label(labels, "a.jpg")
    template = json.loads((labels / "train/a.json").read_text())
    paths = [str(images / "train/a.jpg")]
    for i in range(5):
        data = dict(template, imagePath=f"{i}.jpg")
        (labels / f"train/{i}.json").write_text(json.dumps(data))
        paths.append(str(images / f"train/{i}.jpg"))
    (labels / "train/duplicate.json").write_text(json.dumps(template))
    plan = attach.build_attach_plan(PathDataset(paths), labels, ["person"], images)
    assert len(plan.to_write) == 6
    assert [row["issue"] for row in plan.issues] == ["sample_id_collision"]


def test_label_field_defaults_and_can_be_selected():
    default_args = attach.parse_args([
        "--dataset-name", "dataset", "--label-dir", ".",
        "--class-names", "person", "--tags", "batch",
    ])
    custom_args = attach.parse_args([
        "--dataset-name", "dataset", "--label-dir", ".",
        "--class-names", "person", "--tags", "batch",
        "--label-field", "ground_truth_detect",
    ])
    assert default_args.label_field == "ground_truth"
    assert custom_args.label_field == "ground_truth_detect"


def test_existing_boxes_read_in_batches(monkeypatch):
    monkeypatch.setattr(attach, "QUERY_BATCH_SIZE", 2)
    calls = []

    class Dataset:
        def has_field(self, name):
            return True

        def select(self, ids):
            calls.append(ids)

            class View:
                def values(self, fields):
                    assert fields == ["id", attach.DEFAULT_LABEL_FIELD]
                    return ids, [None] * len(ids)

            return View()

    assert attach.load_existing_boxes(Dataset(), ["a", "b", "c"]) == dict.fromkeys(["a", "b", "c"])
    assert calls == [["a", "b"], ["c"]]
