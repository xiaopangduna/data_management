import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "update_tags", Path(__file__).parents[1] / "scripts/update_tags.py"
)
update = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update)


def test_matching_duplicates_ambiguity_and_missing_hash(tmp_path):
    images = []
    for name, data in [("renamed.jpg", b"a"), ("copy.PNG", b"a"), ("dup.jpg", b"b"), ("unknown.jpg", b"c")]:
        path = tmp_path / name
        path.write_bytes(data)
        images.append(path)
    def sha(data):
        return hashlib.sha256(data).hexdigest()
    rows = [("1", sha(b"a"), ["old"]), ("2", sha(b"b"), []), ("3", sha(b"b"), []), ("4", None, [])]
    updates, issues, matched = update.build_plan(rows, images, ["review"])
    assert updates == ["1"] and matched == 1
    assert [item["issue"] for item in issues] == ["missing_sha256", "ambiguous_sha256", "unmatched"]


def test_scan_and_read_failure(tmp_path):
    (tmp_path / "a.JPG").write_bytes(b"image")
    (tmp_path / "a.txt").write_text("ignored")
    child = tmp_path / "child"
    child.mkdir()
    (child / "b.png").write_bytes(b"image")
    assert len(update.list_images(tmp_path, False)) == 1
    assert len(update.list_images(tmp_path, True)) == 2
    updates, issues, matched = update.build_plan([], [tmp_path / "missing.jpg"], ["review"])
    assert not updates and matched == 0
    assert issues[0]["issue"] == "read_error"


def test_cli():
    args = update.parse_args(["--dataset-name", "demo", "--images-dir", "/images", "--tags", " review,old,review, "])
    assert args.tags == ["review", "old"]
    assert not args.recursive and not args.dry_run
    with pytest.raises(SystemExit):
        update.parse_args(["--dataset-name", "demo", "--images-dir", "/images", "--tags", " , "])


class Sample:
    def __init__(self):
        self.tags = ["old"]
        self.saves = 0

    def save(self):
        self.saves += 1


class Dataset:
    def __init__(self):
        self.sample = Sample()

    def get_field_schema(self):
        return {"sha256": object()}

    def values(self, fields):
        assert fields == ["id", "sha256", "tags"]
        return [["1"], [hashlib.sha256(b"image").hexdigest()], [self.sample.tags]]

    def __getitem__(self, sample_id):
        assert sample_id == "1"
        return self.sample


def test_dry_run_preservation_and_idempotence(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "renamed.jpg").write_bytes(b"image")
    dataset = Dataset()
    fo = SimpleNamespace(dataset_exists=lambda name: True, load_dataset=lambda name: dataset)
    args = update.parse_args(["--dataset-name", "demo", "--images-dir", str(tmp_path), "--tags", "review,old", "--dry-run"])
    assert update.run(args, fo) == 0
    assert dataset.sample.tags == ["old"] and dataset.sample.saves == 0
    assert (tmp_path / "tmp/update_tags_demo.csv").is_file()
    assert "to_update=1" in capsys.readouterr().out
    args.dry_run = False
    assert update.run(args, fo) == 0
    assert dataset.sample.tags == ["old", "review"] and dataset.sample.saves == 1
    assert update.run(args, fo) == 0
    assert dataset.sample.saves == 1
    assert "unchanged=1" in capsys.readouterr().out


def test_missing_dataset_or_hash_field(tmp_path):
    args = update.parse_args(["--dataset-name", "demo", "--images-dir", str(tmp_path), "--tags", "review"])
    fo = SimpleNamespace(dataset_exists=lambda name: False)
    with pytest.raises(ValueError, match="Dataset does not exist"):
        update.run(args, fo)
    fo.dataset_exists = lambda name: True
    fo.load_dataset = lambda name: SimpleNamespace(get_field_schema=lambda: {})
    with pytest.raises(ValueError, match="update_media.py"):
        update.run(args, fo)
