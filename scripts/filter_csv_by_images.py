#!/usr/bin/env python3
"""Filter CSV rows according to file stems without modifying source files."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据文件 stem 删除或保留 CSV 行（不会修改源文件）。"
    )
    parser.add_argument("source_path", type=Path, help="单个文件或文件目录")
    parser.add_argument("csv_path", type=Path, help="待处理 CSV 文件")
    parser.add_argument(
        "action",
        choices=("delete", "keep"),
        help="delete=删除匹配行；keep=只保留匹配行",
    )
    parser.add_argument(
        "--column", default="filename", help="保存文件名的 CSV 列名（默认：filename）"
    )
    parser.add_argument("--recursive", action="store_true", help="递归扫描文件目录")
    parser.add_argument("--output", type=Path, help="输出路径；默认在原 CSV 旁生成新文件")
    parser.add_argument(
        "--overwrite", action="store_true", help="覆盖原 CSV（不能与 --output 同时使用）"
    )
    return parser.parse_args()


def collect_file_stems(source_path: Path, recursive: bool) -> set[str]:
    if not source_path.exists():
        raise FileNotFoundError(f"文件路径不存在：{source_path}")

    if source_path.is_file():
        candidates = [source_path]
    elif source_path.is_dir():
        candidates = source_path.rglob("*") if recursive else source_path.iterdir()
    else:
        raise ValueError(f"路径既不是文件也不是目录：{source_path}")

    return {path.stem for path in candidates if path.is_file()}


def csv_value_stem(value: str) -> str:
    # CSV 单元格可以是文件名，也可以是完整路径；统一比较 stem。
    filename = Path(value.strip().replace("\\", "/")).name
    return Path(filename).stem


def default_output_path(csv_path: Path, action: str) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_{action}{csv_path.suffix}")


def filter_csv(
    csv_path: Path,
    output_path: Path,
    file_stems: set[str],
    column: str,
    action: str,
) -> tuple[int, int]:
    csv_is_empty = not csv_path.exists() or csv_path.stat().st_size == 0
    if csv_is_empty:
        if action == "delete":
            raise ValueError("delete 模式要求 CSV 已存在且包含表头")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=[column])
            writer.writeheader()
            writer.writerows({column: stem} for stem in sorted(file_stems))
        return 0, len(file_stems)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        sample = source.read(8192)
        source.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(source, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError("CSV 缺少表头")
        if column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(f"CSV 中没有列 {column!r}；现有列：{available}")

        rows = list(reader)

    kept_rows = []
    matched = 0
    for row in rows:
        is_match = csv_value_stem(row[column]) in file_stems
        matched += int(is_match)
        if (action == "keep" and is_match) or (action == "delete" and not is_match):
            kept_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=reader.fieldnames,
                dialect=dialect,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(kept_rows)
        os.replace(temporary_name, output_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    return len(rows), matched


def main() -> None:
    args = parse_args()
    if args.overwrite and args.output:
        raise SystemExit("错误：--overwrite 与 --output 不能同时使用")

    csv_path = args.csv_path.resolve()
    output_path = (
        csv_path
        if args.overwrite
        else (args.output.resolve() if args.output else default_output_path(csv_path, args.action))
    )
    if output_path == csv_path and not args.overwrite:
        raise SystemExit("错误：输出路径与输入 CSV 相同；如需覆盖请使用 --overwrite")

    try:
        file_stems = collect_file_stems(args.source_path, args.recursive)
        total, matched = filter_csv(
            csv_path,
            output_path,
            file_stems,
            args.column,
            args.action,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"错误：{error}") from error

    kept = matched if args.action == "keep" else total - matched
    print(f"找到文件 stem：{len(file_stems)} 个")
    print(f"CSV 原始行数：{total}，匹配行数：{matched}，输出行数：{kept}")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()
