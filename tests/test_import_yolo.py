"""Unit tests for YOLO leaf-directory import helpers (no FiftyOne DB)."""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    module_name = name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


import_yolo = load_script("import_yolo.py")


def test_parse_args_requires_class_names_with_labels_dir():
    with pytest.raises(SystemExit):
        import_yolo.parse_args(
            [
                "--dataset-name",
                "demo",
                "--images-dir",
                "/tmp/images",
                "--tags",
                "head,train",
                "--labels-dir",
                "/tmp/labels",
            ]
        )


def test_parse_args_rejects_class_names_without_labels_dir():
    with pytest.raises(SystemExit):
        import_yolo.parse_args(
            [
                "--dataset-name",
                "demo",
                "--images-dir",
                "/tmp/images",
                "--tags",
                "head,train",
                "--class-names",
                "baby_head",
            ]
        )


def test_parse_args_images_only():
    args = import_yolo.parse_args(
        [
            "--dataset-name",
            "demo",
            "--images-dir",
            "/tmp/images",
            "--tags",
            "head,train,train",
        ]
    )
    assert args.tags == ["head", "train"]
    assert args.labels_dir is None


def test_parse_yolo_txt_valid(tmp_path: Path):
    path = tmp_path / "foo.txt"
    path.write_text("0 0.5 0.5 0.2 0.4\n1 0.3 0.3 0.1 0.1\n", encoding="utf-8")
    boxes, error = import_yolo.parse_yolo_txt(path, ["baby_head", "adult_head"])
    assert error is None
    assert boxes is not None
    by_name = {name: bbox for name, bbox in boxes}
    assert by_name["baby_head"] == [0.4, 0.3, 0.2, 0.4]
    assert by_name["adult_head"] == [0.25, 0.25, 0.1, 0.1]


def test_parse_yolo_txt_empty_is_not_an_error(tmp_path: Path):
    path = tmp_path / "foo.txt"
    path.write_text("\n", encoding="utf-8")
    boxes, error = import_yolo.parse_yolo_txt(path, ["baby_head"])
    assert boxes is None
    assert error is None


def test_parse_yolo_txt_class_out_of_range(tmp_path: Path):
    path = tmp_path / "foo.txt"
    path.write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    boxes, error = import_yolo.parse_yolo_txt(path, ["baby_head"])
    assert boxes is None
    assert error is not None


def test_list_leaf_files_ignores_nested(tmp_path: Path):
    leaf = tmp_path / "images" / "train"
    nested = leaf / "nested"
    nested.mkdir(parents=True)
    (leaf / "a.jpg").write_bytes(b"x")
    (nested / "b.jpg").write_bytes(b"y")
    files, subdirs = import_yolo.list_leaf_files(leaf, import_yolo.IMAGE_SUFFIXES)
    assert [path.name for path in files] == ["a.jpg"]
    assert subdirs == 1


def test_scan_images_skips_nested_and_uses_absolute_path(tmp_path: Path):
    leaf = tmp_path / "images" / "train"
    nested = leaf / "nested"
    nested.mkdir(parents=True)
    image = leaf / "a.jpg"
    image.write_bytes(b"x")
    (nested / "b.jpg").write_bytes(b"y")
    items, subdirs = import_yolo.scan_images(leaf)
    assert subdirs == 1
    assert items == [(str(image), "a.jpg", "a")]


def test_scan_images_resolves_symlink(tmp_path: Path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    target = real_dir / "a.jpg"
    target.write_bytes(b"x")
    link_dir = tmp_path / "links"
    link_dir.mkdir()
    (link_dir / "a.jpg").symlink_to(target)
    items, subdirs = import_yolo.scan_images(link_dir)
    assert subdirs == 0
    assert items == [(str(target.resolve()), "a.jpg", "a")]


def test_index_labels_maps_stem(tmp_path: Path):
    labels = tmp_path / "labels"
    labels.mkdir()
    path = labels / "a.txt"
    path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "nested").mkdir()
    label_by_stem, subdirs = import_yolo.index_labels(labels)
    assert subdirs == 1
    assert label_by_stem == {"a": str(path)}


def test_parse_yolo_txt_accepts_str_path(tmp_path: Path):
    path = tmp_path / "foo.txt"
    path.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    boxes, error = import_yolo.parse_yolo_txt(str(path), ["baby_head"])
    assert error is None
    assert boxes == [("baby_head", [0.4, 0.3, 0.2, 0.4])]


def test_parse_yolo_txt_many_chunked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(import_yolo, "PARSE_CHUNK_SIZE", 2)
    monkeypatch.setattr(import_yolo, "PARSE_WORKERS", 2)
    paths = []
    for index in range(5):
        path = tmp_path / f"{index}.txt"
        path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        paths.append(str(path))
    results = import_yolo.parse_yolo_txt_many(paths, ["baby_head"])
    assert len(results) == 5
    assert all(error is None and boxes is not None for boxes, error in results)


def test_tag_list_rejects_blank():
    with pytest.raises(argparse.ArgumentTypeError):
        import_yolo.tag_list("  ,  ")
