"""X-AnyLabeling export/attach helpers without requiring a live dataset write."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


export = load_script("export_xlabel.py")
attach = load_script("attach_xlabel_labels.py")


def test_fo_box_round_trip():
    points = export.fo_box_to_points([0.1, 0.2, 0.3, 0.4], 100, 50)
    assert points == [[10.0, 10.0], [40.0, 30.0]]
    pixel = attach.points_to_pixel_box(points)
    fo_box = attach.pixel_box_to_fo(pixel, 100, 50)
    assert fo_box == (0.1, 0.2, 0.3, 0.4)


def test_polygon_aabb_and_unsupported(tmp_path):
    document = {
        "sample_id": "abc",
        "imageWidth": 10,
        "imageHeight": 10,
        "shapes": [
            {
                "label": "baby_head",
                "shape_type": "polygon",
                "points": [[1, 2], [5, 2], [5, 8], [1, 8]],
            }
        ],
    }
    path = tmp_path / "poly.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    parsed, issues = attach.parse_xlabel_file(path, tmp_path, {"baby_head"})
    assert not issues
    assert parsed.boxes == (("baby_head", (0.1, 0.2, 0.4, 0.6)),)

    document["shapes"][0]["shape_type"] = "rotation"
    path.write_text(json.dumps(document), encoding="utf-8")
    parsed, issues = attach.parse_xlabel_file(path, tmp_path, {"baby_head"})
    assert parsed is None
    assert issues[0]["issue"] == "unsupported_shape"


def test_parse_sample_id_from_description():
    assert attach.parse_fo_sample_id({"description": "fo_sample_id=xyz extra"}) == "xyz"
    assert attach.parse_fo_sample_id({"sample_id": "top"}) == "top"


def test_export_item_json_contains_sample_id():
    item = export.ExportItem(
        sample_id="sid1",
        filepath=Path("/tmp/a.jpg"),
        relpath="sub/a.jpg",
        json_relpath="sub/a.json",
        width=20,
        height=10,
        shapes=[],
    )
    document = export.build_xlabel_document(item)
    assert document["sample_id"] == "sid1"
    assert document["description"] == "fo_sample_id=sid1"
    assert document["imagePath"] == "a.jpg"
    assert document["imageData"] is None


def test_detections_skip_unknown_and_bad_box():
    detections = SimpleNamespace(
        detections=[
            SimpleNamespace(label="baby_head", bounding_box=[0.0, 0.0, 0.5, 0.5]),
            SimpleNamespace(label="other", bounding_box=[0.0, 0.0, 0.2, 0.2]),
            SimpleNamespace(label="baby_head", bounding_box=[0.0, 0.0, 0.0, 0.1]),
        ]
    )
    shapes, unknown = export.detections_to_shapes(detections, 100, 100, {"baby_head"})
    assert unknown == ["other"]
    assert len(shapes) == 1
    assert shapes[0]["shape_type"] == "rectangle"


def test_ensure_symlink_and_apply_export(tmp_path):
    source = tmp_path / "orig.jpg"
    source.write_bytes(b"jpg")
    item = export.ExportItem(
        sample_id="sid1",
        filepath=source,
        relpath="nested/orig.jpg",
        json_relpath="nested/orig.json",
        width=2,
        height=2,
        shapes=[],
    )
    out_dir = tmp_path / "task"
    images, json_files, issues = export.apply_export(out_dir, [item])
    assert images == 1 and json_files == 1 and not issues
    dest = out_dir / "nested" / "orig.jpg"
    assert dest.is_symlink()
    assert dest.resolve() == source.resolve()
    payload = json.loads((out_dir / "nested" / "orig.json").read_text(encoding="utf-8"))
    assert payload["sample_id"] == "sid1"
    manifest = (out_dir / "manifest.csv").read_text(encoding="utf-8")
    assert "sid1" in manifest
    images, json_files, issues = export.apply_export(out_dir, [item])
    assert images == 1 and not issues


def test_attach_parse_unknown_label(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps(
            {
                "sample_id": "sid",
                "imageWidth": 10,
                "imageHeight": 10,
                "shapes": [
                    {
                        "label": "nope",
                        "shape_type": "rectangle",
                        "points": [[0, 0], [2, 2]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    parsed, issues = attach.parse_xlabel_file(path, tmp_path, {"baby_head"})
    assert parsed is None
    assert issues[0]["issue"] == "unknown_label"


def test_cli_requires_include_tags():
    with pytest.raises(SystemExit):
        export.parse_args(["--dataset-name", "x", "--out-dir", "/tmp/out"])
    args = export.parse_args(
        ["--dataset-name", "x", "--out-dir", "/tmp/out", "--include-tags", "relabel"]
    )
    assert args.include_tags == ["relabel"]
