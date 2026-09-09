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


def test_tag_list_rejects_blank():
    with pytest.raises(argparse.ArgumentTypeError):
        import_yolo.tag_list("  ,  ")
