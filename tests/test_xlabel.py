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


attach = load_script("attach_xlabel_labels.py")


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
