import csv
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "extract_by_name", Path(__file__).parents[1] / "scripts/extract_by_name.py"
)
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def write_file(path: Path, data: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)
    return path


def read_report(tmp_path: Path) -> list[dict[str, str]]:
    path = tmp_path / "tmp" / "extract_by_name.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cli_defaults():
    args = extract.parse_args(
        ["--names-dir", "/a", "--source-dir", "/b", "--source-dir", "/c", "--out-dir", "/out"]
    )
    assert args.match == "stem"
    assert args.export_media == "copy"
    assert args.source_dirs == [Path("/b"), Path("/c")]
    assert not args.flatten and not args.recursive and not args.dry_run


def test_stem_match_copies_into_source_subdirs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    names = tmp_path / "A"
    source_b = tmp_path / "B"
    source_c = tmp_path / "C"
    source_d = tmp_path / "D"
    out = tmp_path / "out"
    write_file(names / "foo.jpg")
    write_file(names / "bar.PNG")
    write_file(names / "missing.jpg")
    write_file(source_b / "foo.jpg", "image")
    write_file(source_b / "extra.jpg", "skip")
    write_file(source_c / "foo.txt", "label")
    write_file(source_c / "bar.json", "json")
    write_file(source_d / "other.jpg", "no")
    args = extract.parse_args(
        [
            "--names-dir",
            str(names),
            "--source-dir",
            str(source_b),
            "--source-dir",
            str(source_c),
            "--source-dir",
            str(source_d),
            "--out-dir",
            str(out),
        ]
    )
    assert extract.run(args) == 2
    assert (out / "B" / "foo.jpg").read_text() == "image"
    assert (out / "C" / "foo.txt").read_text() == "label"
    assert (out / "C" / "bar.json").read_text() == "json"
    assert not (out / "B" / "extra.jpg").exists()
    assert not (out / "D" / "other.jpg").exists()
    rows = read_report(tmp_path)
    assert {row["status"] for row in rows} == {"copied", "missing"}
    assert any(row["status"] == "missing" and row["name"] == "missing" for row in rows)
    output = capsys.readouterr().out
    assert "copied=3" in output
    assert "written=3" in output


def test_name_match_requires_same_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = tmp_path / "A"
    source = tmp_path / "B"
    out = tmp_path / "out"
    write_file(names / "foo.jpg")
    write_file(source / "foo.txt")
    write_file(source / "foo.JPG", "hit")
    args = extract.parse_args(
        [
            "--names-dir",
            str(names),
            "--source-dir",
            str(source),
            "--out-dir",
            str(out),
            "--match",
            "name",
        ]
    )
    assert extract.run(args) == 0
    assert (out / "B" / "foo.JPG").read_text() == "hit"
    assert not (out / "B" / "foo.txt").exists()


def test_flatten_collision_skips_all_and_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = tmp_path / "A"
    source_b = tmp_path / "B"
    source_c = tmp_path / "C"
    out = tmp_path / "out"
    write_file(names / "foo.jpg")
    write_file(source_b / "foo.jpg", "b")
    write_file(source_c / "foo.jpg", "c")
    args = extract.parse_args(
        [
            "--names-dir",
            str(names),
            "--source-dir",
            str(source_b),
            "--source-dir",
            str(source_c),
            "--out-dir",
            str(out),
            "--flatten",
            "--dry-run",
        ]
    )
    assert extract.run(args) == 2
    assert not out.exists()
    rows = read_report(tmp_path)
    assert len(rows) == 2
    assert all(row["status"] == "collision" for row in rows)


def test_recursive_flattens_relative_path_and_detects_same_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = tmp_path / "A"
    source = tmp_path / "B"
    out = tmp_path / "out"
    write_file(names / "foo.jpg")
    write_file(source / "nested" / "foo.jpg", "nested")
    args = extract.parse_args(
        ["--names-dir", str(names), "--source-dir", str(source), "--out-dir", str(out)]
    )
    assert extract.run(args) == 2
    assert not (out / "B" / "foo.jpg").exists()
    args.recursive = True
    assert extract.run(args) == 0
    assert (out / "B" / "foo.jpg").read_text() == "nested"

    write_file(source / "other" / "foo.jpg", "other")
    out2 = tmp_path / "out2"
    args.out_dir = out2
    assert extract.run(args) == 2
    assert not out2.exists()
    assert any(row["status"] == "collision" for row in read_report(tmp_path))


def test_existing_destination_is_collision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = tmp_path / "A"
    source = tmp_path / "B"
    out = tmp_path / "out"
    write_file(names / "foo.jpg")
    write_file(source / "foo.jpg", "new")
    existing = write_file(out / "B" / "foo.jpg", "old")
    args = extract.parse_args(
        ["--names-dir", str(names), "--source-dir", str(source), "--out-dir", str(out)]
    )
    assert extract.run(args) == 2
    assert existing.read_text() == "old"


def test_symlink_and_duplicate_source_basename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = tmp_path / "A"
    source = tmp_path / "B"
    out = tmp_path / "out"
    write_file(names / "foo.jpg")
    original = write_file(source / "foo.jpg", "image")
    args = extract.parse_args(
        [
            "--names-dir",
            str(names),
            "--source-dir",
            str(source),
            "--out-dir",
            str(out),
            "--export-media",
            "symlink",
        ]
    )
    assert extract.run(args) == 0
    dest = out / "B" / "foo.jpg"
    assert dest.is_symlink()
    assert dest.resolve() == original.resolve()
    assert dest.read_text() == "image"

    other = tmp_path / "other" / "B"
    write_file(other / "bar.jpg")
    with pytest.raises(ValueError, match="share the same name"):
        extract.build_plan(
            {"foo": names / "foo.jpg"},
            [source, other],
            out,
            recursive=False,
            mode="stem",
            flatten=False,
        )


def test_duplicate_names_dir_stem_still_extracts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = tmp_path / "A"
    source = tmp_path / "B"
    out = tmp_path / "out"
    write_file(names / "foo.jpg")
    write_file(names / "foo.png")
    write_file(source / "foo.txt", "label")
    args = extract.parse_args(
        ["--names-dir", str(names), "--source-dir", str(source), "--out-dir", str(out)]
    )
    assert extract.run(args) == 0
    assert (out / "B" / "foo.txt").read_text() == "label"
    rows = read_report(tmp_path)
    assert any(row["status"] == "skipped" for row in rows)
