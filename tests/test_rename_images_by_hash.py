import hashlib
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "rename_images_by_hash", Path(__file__).parents[1] / "scripts/rename_images_by_hash.py"
)
rename = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rename
spec.loader.exec_module(rename)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_rename_and_keep_identical_duplicates(tmp_path: Path):
    (tmp_path / "original.JPG").write_bytes(b"same")
    (tmp_path / "another.jpg").write_bytes(b"same")
    (tmp_path / "other.png").write_bytes(b"other")
    (tmp_path / "note.txt").write_text("leave me")
    plan = rename.build_plan(rename.scan_images(tmp_path, recursive=False))
    rename.apply_plan(plan)
    assert (tmp_path / f"{digest(b'same')}.jpg").read_bytes() == b"same"
    assert (tmp_path / f"{digest(b'same')}-2.jpg").read_bytes() == b"same"
    assert (tmp_path / f"{digest(b'other')}.png").read_bytes() == b"other"
    assert (tmp_path / "note.txt").read_text() == "leave me"


def test_recursive_and_second_run_is_stable(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "image.webp").write_bytes(b"image")
    assert rename.scan_images(tmp_path, recursive=False) == []
    plan = rename.build_plan(rename.scan_images(tmp_path, recursive=True))
    rename.apply_plan(plan)
    assert len(plan) == 1
    assert rename.build_plan(rename.scan_images(tmp_path, recursive=True)) == []
