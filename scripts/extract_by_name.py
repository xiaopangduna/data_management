"""Extract files from source directories whose names match a names directory."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

CSV_COLUMNS = ("status", "name", "source", "dest", "detail")
STATUS_ORDER = {"collision": 0, "missing": 1, "skipped": 2, "copied": 3}


def nonempty_path(value: str) -> Path:
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return Path(stripped)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names-dir", required=True, type=nonempty_path, help="名单目录")
    parser.add_argument(
        "--source-dir",
        required=True,
        action="append",
        type=nonempty_path,
        dest="source_dirs",
        help="源目录，可重复",
    )
    parser.add_argument("--out-dir", required=True, type=nonempty_path, help="输出根目录")
    parser.add_argument(
        "--match",
        choices=("stem", "name"),
        default="stem",
        help="stem=不含扩展名（默认）；name=文件名且扩展名大小写不敏感",
    )
    parser.add_argument(
        "--export-media",
        choices=("copy", "symlink"),
        default="copy",
        help="复制文件（默认）或创建指向原文件的软链",
    )
    parser.add_argument("--flatten", action="store_true", help="命中文件直接写入 --out-dir")
    parser.add_argument("--recursive", action="store_true", help="递归扫描名单目录和源目录")
    parser.add_argument("--dry-run", action="store_true", help="只写报告，不复制或创建软链")
    return parser.parse_args(argv)


def match_key(path: Path, mode: str) -> str:
    if mode == "stem":
        return path.stem
    return f"{path.stem}{path.suffix.lower()}"


def list_files(root: Path, recursive: bool) -> list[Path]:
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in candidates if path.is_file())


def require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def collect_name_keys(
    names_dir: Path, recursive: bool, mode: str
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    keys: dict[str, Path] = {}
    rows: list[dict[str, str]] = []
    for path in list_files(names_dir, recursive):
        key = match_key(path, mode)
        if key in keys:
            rows.append(
                row("skipped", key, str(path), detail=f"duplicate name in names-dir; keep {keys[key]}")
            )
            continue
        keys[key] = path
    return keys, rows


def destination_for(out_dir: Path, source_root: Path, path: Path, flatten: bool) -> Path:
    if flatten:
        return out_dir / path.name
    return out_dir / source_root.name / path.name


def row(
    status: str,
    name: str,
    source: str = "",
    dest: str = "",
    detail: str = "",
) -> dict[str, str]:
    return dict(status=status, name=name, source=source, dest=dest, detail=detail)


def build_plan(
    name_keys: dict[str, Path],
    source_dirs: list[Path],
    out_dir: Path,
    recursive: bool,
    mode: str,
    flatten: bool,
) -> tuple[list[tuple[Path, Path]], list[dict[str, str]]]:
    if not flatten:
        labels = [path.name for path in source_dirs]
        if len(set(labels)) != len(labels):
            raise ValueError(
                "source directories share the same name; rename a parent or use --flatten: "
                + ", ".join(labels)
            )

    by_dest: dict[Path, list[tuple[str, Path]]] = defaultdict(list)
    for source_root in source_dirs:
        for path in list_files(source_root, recursive):
            key = match_key(path, mode)
            if key not in name_keys:
                continue
            dest = destination_for(out_dir, source_root, path, flatten)
            by_dest[dest].append((key, path))

    to_copy: list[tuple[Path, Path]] = []
    rows: list[dict[str, str]] = []
    matched: set[str] = set()
    for dest, items in sorted(by_dest.items(), key=lambda item: str(item[0])):
        sources = [str(path) for _, path in items]
        if len(items) > 1:
            detail = "multiple sources: " + ", ".join(sources)
            for key, path in items:
                matched.add(key)
                rows.append(row("collision", key, str(path), str(dest), detail))
            continue
        key, path = items[0]
        matched.add(key)
        if dest.exists() or dest.is_symlink():
            rows.append(row("collision", key, str(path), str(dest), "destination exists"))
            continue
        rows.append(row("copied", key, str(path), str(dest)))
        to_copy.append((path, dest))

    searched = ", ".join(path.name for path in source_dirs)
    for key, origin in sorted(name_keys.items()):
        if key not in matched:
            rows.append(row("missing", key, str(origin), detail=f"not found in {searched}"))
    return to_copy, rows


def write_report(rows: list[dict[str, str]]) -> Path:
    output = Path.cwd() / "tmp" / "extract_by_name.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda item: (STATUS_ORDER.get(item["status"], 9), item["name"], item["source"], item["dest"]),
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)
    return output


def apply_plan(to_copy: list[tuple[Path, Path]], export_media: str) -> None:
    for source, dest in to_copy:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f"destination exists: {dest}")
        if export_media == "copy":
            shutil.copy2(source, dest)
        else:
            dest.symlink_to(source.resolve())


def count_status(rows: list[dict[str, str]], status: str) -> int:
    return sum(item["status"] == status for item in rows)


def run(args: argparse.Namespace) -> int:
    names_dir = require_dir(args.names_dir, "names-dir")
    source_dirs = list(dict.fromkeys(require_dir(path, "source-dir") for path in args.source_dirs))
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"out-dir is not a directory: {out_dir}")

    name_keys, extra_rows = collect_name_keys(names_dir, args.recursive, args.match)
    to_copy, rows = build_plan(
        name_keys, source_dirs, out_dir, args.recursive, args.match, args.flatten
    )
    rows = extra_rows + rows
    report = write_report(rows)
    copied = count_status(rows, "copied")
    missing = count_status(rows, "missing")
    collision = count_status(rows, "collision")
    skipped = count_status(rows, "skipped")

    print(f"mode={'dry-run' if args.dry_run else 'extract'}")
    print(f"match={args.match}")
    print(f"names={len(name_keys)}")
    print(f"sources={len(source_dirs)}")
    print(f"copied={copied}")
    print(f"missing={missing}")
    print(f"collision={collision}")
    print(f"skipped={skipped}")
    print(f"csv_path={report}")
    if not args.dry_run:
        apply_plan(to_copy, args.export_media)
        print(f"written={len(to_copy)}")
    return 2 if missing or collision else 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
