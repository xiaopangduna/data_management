"""Copy a dataset to a new directory, materializing file symlinks. Use --dry-run to preview."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import stat
import sys
import time

DEFAULT_DIRECTORY = Path('/mnt/nvme_data/data/head_train_data/baby_head_adult_head')


def materialize(link: Path, destination: Path) -> None:
    """Copy a regular file or file symlink without modifying the source."""
    shutil.copy2(link, destination, follow_symlinks=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument('--out-dir', required=True, type=Path, help='New output directory (must not exist)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without creating output')
    parser.add_argument('--limit', type=int, help='Process at most N files for a trial copy')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be greater than zero')
    root = args.directory.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f'Not a directory: {root}')
    output = args.out_dir.expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError(f'Output already exists: {output}')
    output = output.resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('Input and output directories must not overlap')
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    scanned = completed = total = errors = 0
    started = last_report = time.monotonic()
    mode = 'PREVIEW' if args.dry_run else 'COPY'

    def report(final: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if final or now - last_report >= 2:
            print(
                f'{mode} scanned={scanned} completed={completed} '
                f'bytes={total} errors={errors} elapsed={now - started:.1f}s',
                flush=True,
            )
            last_report = now

    def entries(directory: Path):
        # scandir yields entries immediately, without listing/sorting a whole directory.
        with os.scandir(directory) as iterator:
            for entry in iterator:
                report()
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    if not args.dry_run:
                        (output / path.relative_to(root)).mkdir()
                    yield from entries(path)
                else:
                    yield path

    print(f'{mode} source={root} output={output} output_free_bytes={free}', flush=True)
    stream = None
    try:
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=False)
        stream = entries(root)
        for path in stream:
            scanned += 1
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f'Not a regular file (directory symlinks unsupported): {path}')
            if not args.dry_run:
                materialize(path, output / path.relative_to(root))
                completed += 1
            total += info.st_size
            report()
            if args.limit is not None and scanned >= args.limit:
                break
    except (OSError, ValueError, RuntimeError) as error:
        errors += 1
        print(f'ERROR: {error}', file=sys.stderr)
        if not args.dry_run:
            print('Stopped; partial output retained. Use a new output directory to retry.', file=sys.stderr)
    finally:
        if stream is not None:
            stream.close()
        if not args.dry_run:
            print('Syncing written files to disk...', flush=True)
            os.sync()
        report(final=True)
    if args.dry_run:
        print('Preview only; no output created. bytes is the estimated copy size.')
        if total > free:
            print('WARNING: Estimated copy size exceeds available output space.', file=sys.stderr)
    return int(errors > 0)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('Interrupted; partial output retained. Use a new output directory for another run.', file=sys.stderr)
        sys.exit(130)
