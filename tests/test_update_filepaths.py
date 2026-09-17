import hashlib
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "update_filepaths", Path(__file__).parents[1] / "scripts/update_filepaths.py"
)
update = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = update
spec.loader.exec_module(update)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_cli_is_dataset_images_dir_and_optional_dry_run():
    args = update.parse_args(
        ["--dataset", "demo", "--images-dir", "/images", "--dry-run"]
    )
    assert args.dataset == "demo"
    assert args.images_dir == Path("/images")
    assert args.dry_run


def test_updates_matching_hash_and_ignores_dataset_remainder(tmp_path: Path):
    image = tmp_path / "renamed.jpg"
    image.write_bytes(b"image")
    by_hash = update.hash_index(update.list_images(tmp_path))
    rows = [
        ("1", str(tmp_path / "old.jpg"), sha(b"image")),
        ("2", str(tmp_path / "outside.jpg"), sha(b"not-in-directory")),
        ("3", str(tmp_path / "no-hash.jpg"), None),
    ]
    updates, issues, unchanged, ignored = update.build_plan(rows, by_hash)
    assert updates == [update.PathUpdate("1", str(tmp_path / "old.jpg"), str(image))]
    assert not issues and unchanged == 0 and ignored == 2


def test_existing_path_is_unchanged_even_when_content_is_duplicated(tmp_path: Path):
    first = tmp_path / "image.jpg"
    second = tmp_path / "val_image.jpg"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    by_hash = update.hash_index(update.list_images(tmp_path))
    rows = [("1", str(first), sha(b"same")), ("2", str(second), sha(b"same"))]
    updates, issues, unchanged, ignored = update.build_plan(rows, by_hash)
    assert not updates and not issues
    assert unchanged == 2 and ignored == 0


def test_duplicate_hash_uses_exact_old_basename(tmp_path: Path):
    first = tmp_path / "image.jpg"
    second = tmp_path / "val_image.jpg"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    by_hash = update.hash_index(update.list_images(tmp_path))
    old = tmp_path / "old" / "val_image.jpg"
    updates, issues, unchanged, ignored = update.build_plan(
        [("1", str(old), sha(b"same"))], by_hash
    )
    assert updates == [update.PathUpdate("1", str(old), str(second))]
    assert not issues and unchanged == 0 and ignored == 0


def test_duplicate_hash_without_name_match_is_ambiguous(tmp_path: Path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    by_hash = update.hash_index(update.list_images(tmp_path))
    updates, issues, unchanged, ignored = update.build_plan(
        [("1", str(tmp_path / "old.jpg"), sha(b"same"))], by_hash
    )
    assert not updates and unchanged == 0 and ignored == 0
    assert [row["issue"] for row in issues] == ["ambiguous_image_hash"]


def test_multiple_samples_for_one_target_is_rejected(tmp_path: Path):
    target = tmp_path / "target.jpg"
    target.write_bytes(b"same")
    by_hash = {sha(b"same"): [str(target)]}
    rows = [
        ("1", str(tmp_path / "old-1.jpg"), sha(b"same")),
        ("2", str(tmp_path / "old-2.jpg"), sha(b"same")),
    ]
    updates, issues, unchanged, ignored = update.build_plan(rows, by_hash)
    assert not updates and unchanged == 0 and ignored == 0
    assert [row["issue"] for row in issues] == [
        "sample_hash_collision",
        "sample_hash_collision",
    ]


def test_duplicate_hash_groups_lists_all_paths(tmp_path: Path):
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    unique = tmp_path / "c.jpg"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    unique.write_bytes(b"other")
    by_hash = update.hash_index(update.list_images(tmp_path))
    groups = update.duplicate_hash_groups(by_hash)
    assert groups == [(sha(b"same"), [str(first), str(second)])]


def test_write_duplicates_csv(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    groups = [("abc", ["/tmp/a.jpg", "/tmp/b.jpg"])]
    report = update.write_duplicates("demo", groups)
    assert report == tmp_path / "tmp" / "update_filepaths_demo_duplicates.csv"
    assert report.read_text(encoding="utf-8") == (
        "sha256,count,path\n"
        "abc,2,/tmp/a.jpg\n"
        "abc,2,/tmp/b.jpg\n"
    )
