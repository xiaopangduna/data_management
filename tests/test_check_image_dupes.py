import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "check_image_dupes", Path(__file__).parents[1] / "scripts/check_image_dupes.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_rgb(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)
    return path


def named_sha(path: Path) -> str:
    """Stable digests keyed by stem for classification tests."""
    return sha(path.stem.encode())


def named_phash(path: Path) -> str:
    table = {
        "new": "01" + "a" * 14,
        "exact": "ff" + "a" * 14,
        "near": "03" + "a" * 14,
        "a": "aa" + "a" * 14,
        "b": "aa" + "a" * 14,
        "dup": "05" + "a" * 14,
    }
    return table.get(path.stem, "00" + "a" * 14)


def test_cli_defaults():
    args = check.parse_args(
        ["--dataset", "demo", "--images-dir", "/images", "--out-dir", "/out"]
    )
    assert args.dataset == "demo"
    assert args.images_dir == Path("/images")
    assert args.out_dir == Path("/out")
    assert args.rename_by_hash is True
    assert args.suffixes == check.IMAGE_SUFFIXES
    assert args.recursive is True
    assert not args.dry_run
    folder_only = check.parse_args(["--images-dir", "/images", "--out-dir", "/out"])
    assert folder_only.dataset is None
    base = ["--dataset", "demo", "--images-dir", "/images", "--out-dir", "/out"]
    assert check.parse_args(base + ["--no-recursive"]).recursive is False
    with pytest.raises(SystemExit):
        check.parse_args(base + ["--export-media", "symlink"])
    with pytest.raises(SystemExit):
        check.parse_args(base + ["--materialize", "new"])
    with pytest.raises(SystemExit):
        check.parse_args(base + ["--hashes", "sha256"])


def test_suffixes_subset_and_reject_unknown():
    args = check.parse_args(
        [
            "--dataset",
            "demo",
            "--images-dir",
            "/images",
            "--out-dir",
            "/out",
            "--suffixes",
            ".jpg,.PNG",
        ]
    )
    assert args.suffixes == frozenset({".jpg", ".png"})
    with pytest.raises(SystemExit):
        check.parse_args(
            [
                "--dataset",
                "demo",
                "--images-dir",
                "/images",
                "--out-dir",
                "/out",
                "--suffixes",
                ".txt",
            ]
        )


def test_build_indexes_and_classify(tmp_path):
    images_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    images_dir.mkdir()
    new_path = save_rgb(images_dir / "new.jpg", (1, 0, 0))
    exact_path = save_rgb(images_dir / "exact.jpg", (2, 0, 0))
    near_path = save_rgb(images_dir / "near.jpg", (3, 0, 0))

    by_sha, by_phash, missing_sha, missing_ph = check.build_hash_indexes(
        ["1", "2", "3"],
        [named_sha(exact_path), None, sha(b"other")],
        ["ff" + "a" * 14, "03" + "a" * 14, ""],
        ["/ds/exact.jpg", "/ds/near.jpg", "/ds/other.jpg"],
    )
    assert missing_sha == 1 and missing_ph == 1
    assert len(by_sha[named_sha(exact_path)]) == 1
    assert len(by_phash["03" + "a" * 14]) == 1

    exact_row = check.classify_image(
        exact_path,
        named_sha(exact_path),
        named_phash(exact_path),
        by_sha,
        by_phash,
        out_dir,
        rename_by_hash=True,
        match_dataset=True,
    )
    near_row = check.classify_image(
        near_path,
        named_sha(near_path),
        named_phash(near_path),
        by_sha,
        by_phash,
        out_dir,
        rename_by_hash=True,
        match_dataset=True,
    )
    new_row = check.classify_image(
        new_path,
        named_sha(new_path),
        named_phash(new_path),
        by_sha,
        by_phash,
        out_dir,
        rename_by_hash=True,
        match_dataset=True,
    )
    assert exact_row["status"] == "exact_dup"
    assert exact_row["sample_ids"] == "1"
    assert near_row["status"] == "near_dup"
    assert near_row["sample_ids"] == "2"
    assert new_row["status"] == "new"
    assert new_row["out_filepath"].endswith(f"new/{named_sha(new_path)}.jpg")
    assert new_row["canonical_sha256_name"] == f"{named_sha(new_path)}.jpg"


def test_destination_hash_flat_and_dest_exists(tmp_path):
    out_dir = tmp_path / "out"
    path = save_rgb(tmp_path / "images" / "sub" / "a.jpg", (1, 2, 3))
    digest = "abc123"
    dest = check.destination_for(
        out_dir, path, "new", rename_by_hash=True, sha256=digest
    )
    assert dest == out_dir / "new" / f"{digest}.jpg"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"existing")
    rows = [
        check.empty_row(
            path,
            status="new",
            sha256=digest,
            out_filepath=str(dest),
            canonical_sha256_name=f"{digest}.jpg",
        )
    ]
    to_write, rows, skipped = check.plan_materialize(rows)
    assert not to_write and skipped == 1
    assert rows[0]["issue"] == "dest_exists"


def test_corrupt_truncated_image(tmp_path):
    images_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    images_dir.mkdir()
    good = save_rgb(images_dir / "good.jpg", (10, 20, 30))
    bad = images_dir / "bad.jpg"
    data = good.read_bytes()
    bad.write_bytes(data[: max(40, len(data) // 4)])

    rows = check.hash_and_classify(
        [bad],
        {},
        {},
        out_dir,
        lambda path: sha(path.read_bytes()),
        named_phash,
        rename_by_hash=True,
        match_dataset=True,
    )
    assert rows[0]["status"] == "corrupt"
    assert rows[0]["issue"] == "truncated"
    assert rows[0]["sha256"]
    assert rows[0]["out_filepath"].endswith(f"corrupt/{rows[0]['sha256']}.jpg")


def test_batch_dup_keeps_first_for_dataset_match(tmp_path):
    images_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    first = save_rgb(images_dir / "a.jpg", (9, 9, 9))
    second = images_dir / "b.jpg"
    second.write_bytes(first.read_bytes())
    digest = sha(first.read_bytes())
    rows = check.hash_and_classify(
        [first, second],
        {},
        {},
        out_dir,
        lambda path: sha(path.read_bytes()),
        named_phash,
        rename_by_hash=True,
        match_dataset=True,
    )
    assert rows[0]["status"] == "new"
    assert rows[0]["sha256"] == digest
    assert rows[1]["status"] == "batch_dup"
    assert rows[1]["detail"] == str(first)
    assert rows[1]["out_filepath"].endswith(f"batch_dup/{digest}.jpg")


def test_hash_and_classify_read_error(tmp_path):
    images_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    images_dir.mkdir()
    missing = images_dir / "gone.jpg"

    def boom(_path: Path) -> str:
        raise OSError("no such file")

    rows = check.hash_and_classify(
        [missing], {}, {}, out_dir, boom, boom, rename_by_hash=True, match_dataset=True
    )
    assert rows[0]["status"] == "corrupt"
    assert rows[0]["issue"] == "read_error"
    assert "no such file" in rows[0]["detail"]


def test_ambiguous_sha256_detail(tmp_path):
    out_dir = tmp_path / "out"
    path = save_rgb(tmp_path / "images" / "dup.jpg", (5, 5, 5))
    digest = named_sha(path)
    by_sha = {digest: [("1", "/a"), ("2", "/b")]}
    row = check.classify_image(
        path,
        digest,
        named_phash(path),
        by_sha,
        {},
        out_dir,
        rename_by_hash=True,
        match_dataset=True,
    )
    assert row["status"] == "exact_dup"
    assert row["detail"] == "ambiguous_sha256"
    assert row["issue"] == "ambiguous_sha256"
    assert row["sample_ids"] == "1,2"


class Dataset:
    def __init__(self, ids, sha256s, phashes, filepaths):
        self._ids = ids
        self._sha256s = sha256s
        self._phashes = phashes
        self._filepaths = filepaths

    def get_field_schema(self):
        return {"sha256": object(), "phash": object()}

    def values(self, fields):
        assert fields == ["id", "sha256", "phash", "filepath"]
        return [self._ids, self._sha256s, self._phashes, self._filepaths]


def test_run_copies_and_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    images_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    save_rgb(images_dir / "new.jpg", (1, 0, 0))
    save_rgb(images_dir / "exact.jpg", (2, 0, 0))
    save_rgb(images_dir / "near.jpg", (3, 0, 0))
    dataset = Dataset(
        ["1", "2"],
        [named_sha(images_dir / "exact.jpg"), sha(b"other")],
        ["ff" + "a" * 14, "03" + "a" * 14],
        ["/ds/exact.jpg", "/ds/near.jpg"],
    )
    fo = SimpleNamespace(dataset_exists=lambda name: True, load_dataset=lambda name: dataset)

    dry_args = check.parse_args(
        [
            "--dataset",
            "demo",
            "--images-dir",
            str(images_dir),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ]
    )
    assert check.run(dry_args, fo, named_sha, named_phash) == 0
    assert not any(out_dir.rglob("*")) if not out_dir.exists() else not list(out_dir.rglob("*.jpg"))
    output = capsys.readouterr().out
    assert "new=1" in output and "exact_dup=1" in output and "near_dup=1" in output
    assert "written=0" in output
    report = tmp_path / "tmp" / "check_image_dupes_demo.csv"
    assert report.is_file()

    write_args = check.parse_args(
        ["--dataset", "demo", "--images-dir", str(images_dir), "--out-dir", str(out_dir)]
    )
    assert check.run(write_args, fo, named_sha, named_phash) == 0
    new_dest = out_dir / "new" / f"{named_sha(images_dir / 'new.jpg')}.jpg"
    exact_dest = out_dir / "exact_dup" / f"{named_sha(images_dir / 'exact.jpg')}.jpg"
    near_dest = out_dir / "near_dup" / f"{named_sha(images_dir / 'near.jpg')}.jpg"
    assert new_dest.is_file() and not new_dest.is_symlink()
    assert exact_dest.is_file()
    assert near_dest.is_file()
    assert "written=3" in capsys.readouterr().out


def test_list_images_respects_suffixes(tmp_path):
    images_dir = tmp_path / "images"
    save_rgb(images_dir / "a.jpg", (1, 0, 0))
    save_rgb(images_dir / "b.png", (0, 1, 0))
    (images_dir / "c.txt").write_text("x")
    listed = check.list_images(images_dir, False, frozenset({".jpg"}))
    assert [path.name for path in listed] == ["a.jpg"]


def test_folder_only_mode_no_dataset(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    images_dir = tmp_path / "images"
    out_dir = tmp_path / "out"
    first = save_rgb(images_dir / "a.jpg", (1, 2, 3))
    second = images_dir / "b.jpg"
    second.write_bytes(first.read_bytes())
    save_rgb(images_dir / "c.jpg", (9, 9, 9))
    good = save_rgb(images_dir / "good.jpg", (4, 5, 6))
    bad = images_dir / "bad.jpg"
    bad.write_bytes(good.read_bytes()[: max(40, len(good.read_bytes()) // 4)])

    args = check.parse_args(
        ["--images-dir", str(images_dir), "--out-dir", str(out_dir)]
    )
    assert args.dataset is None
    assert check.run(args, fo=None, compute_sha256=lambda p: sha(p.read_bytes())) == 0
    output = capsys.readouterr().out
    assert "dataset=none" in output
    assert "new=3" in output
    assert "batch_dup=1" in output
    assert "corrupt=1" in output
    assert "exact_dup=0" in output and "near_dup=0" in output
    assert (tmp_path / "tmp" / "check_image_dupes_folder.csv").is_file()
    assert (out_dir / "new").is_dir()
    assert (out_dir / "batch_dup").is_dir()
    assert (out_dir / "corrupt").is_dir()


def test_missing_fields_error(tmp_path):
    class BadDataset:
        def get_field_schema(self):
            return {"sha256": object()}

    fo = SimpleNamespace(
        dataset_exists=lambda name: True, load_dataset=lambda name: BadDataset()
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    args = check.parse_args(
        ["--dataset", "demo", "--images-dir", str(images_dir), "--out-dir", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="update_media.py"):
        check.run(args, fo, named_sha, named_phash)
