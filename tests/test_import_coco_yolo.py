"""Tests for COCO-tree import CLI and split helpers (no FiftyOne DB)."""

import importlib.util
import sys
from pathlib import Path

import pytest

from data_management.yolo_import import (
    COCO_80_CLASSES,
    coco_split_dirs,
    list_split_names,
    prepare_leaf,
)

ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    module_name = name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


import_coco_yolo = load_script("import_coco_yolo.py")


def test_parse_args_requires_coco_root_and_dataset_name():
    with pytest.raises(SystemExit):
        import_coco_yolo.parse_args([])
    with pytest.raises(SystemExit):
        import_coco_yolo.parse_args(["--coco-root", "/tmp/coco"])


def test_parse_args_defaults():
    args = import_coco_yolo.parse_args(
        ["--coco-root", "/tmp/coco", "--dataset-name", "coco2017"]
    )
    assert args.replace is False
    assert args.dry_run is False
    assert not hasattr(args, "class_names")
    assert not hasattr(args, "source_tag")
    assert len(COCO_80_CLASSES) == 80


def test_list_split_names(tmp_path: Path):
    images = tmp_path / "images"
    (images / "train2017").mkdir(parents=True)
    (images / "val2017").mkdir()
    (images / "test2017").mkdir()
    (images / "note.txt").write_text("x", encoding="utf-8")
    assert list_split_names(images) == ["test2017", "train2017", "val2017"]


def test_coco_split_dirs_missing_labels(tmp_path: Path):
    images_dir, labels_dir = coco_split_dirs(tmp_path, "test2017")
    assert images_dir == tmp_path / "images" / "test2017"
    assert labels_dir is None
    (tmp_path / "labels" / "test2017").mkdir(parents=True)
    _, labels_dir = coco_split_dirs(tmp_path, "test2017")
    assert labels_dir == tmp_path / "labels" / "test2017"


def test_prepare_leaf_unlabeled_split_and_skip_existing(tmp_path: Path):
    images = tmp_path / "images" / "test2017"
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"x")
    (images / "b.jpg").write_bytes(b"y")
    result = prepare_leaf(
        images,
        ["coco", "test2017"],
        None,
        None,
        existing={str(images / "a.jpg")},
        dry_run=True,
        relpath_root=tmp_path,
    )
    assert result.skipped == 1
    assert result.unlabeled == 1
    assert result.pending == []
    assert result.new_filepaths == [str(images / "b.jpg")]
