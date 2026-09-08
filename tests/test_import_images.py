"""Image import checks without requiring a running MongoDB."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location("import_images", Path(__file__).parents[1] / "scripts/import_images.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def args(root, *extra):
    return module.parse_args(["--images-root", str(root), "--dataset-name", "images", *extra])


def test_scan_nested_duplicate_and_broken_images(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (2, 2)).save(nested / "image.PNG")
    (tmp_path / "alias.png").symlink_to(nested / "image.PNG")
    (tmp_path / "bad.jpg").write_text("broken")
    (tmp_path / "ignore.txt").write_text("ignored")
    result = module.scan_images(tmp_path, args(tmp_path, "--verify-images"))
    assert result == [((nested / "image.PNG").resolve(), "alias.png")]
    assert len(module.scan_images(tmp_path, args(tmp_path, "--no-recursive", "--extensions", "PNG"))) == 1


def test_dry_run_never_imports_fiftyone(tmp_path, monkeypatch):
    (tmp_path / "image.jpg").write_bytes(b"scan does not decode")
    monkeypatch.setitem(sys.modules, "fiftyone", None)
    assert module.main(["--images-root", str(tmp_path), "--dataset-name", "images", "--dry-run"]) == 0


def test_empty_directory_and_invalid_batch(tmp_path):
    assert module.main(["--images-root", str(tmp_path), "--dataset-name", "images", "--dry-run"]) == 1
    with pytest.raises(SystemExit):
        args(tmp_path, "--batch-size", "0")


def test_existing_dataset_is_never_created(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "fiftyone", SimpleNamespace(dataset_exists=lambda name: True))
    with pytest.raises(ValueError, match="already exists"):
        module.import_images(args(tmp_path), [(tmp_path / "a.jpg", "a.jpg")])


def test_batches_and_partial_failure(tmp_path, monkeypatch, caplog):
    class Dataset:
        def __init__(self, **kwargs):
            assert kwargs == {"name": "images", "persistent": True}
            self.samples = []
            self.calls = 0

        def add_samples(self, batch):
            self.calls += 1
            self.samples.extend(batch)
            if self.calls == 2:
                raise RuntimeError("write failed")

        def __len__(self):
            return len(self.samples)

    dataset = Dataset(name="images", persistent=True)
    monkeypatch.setitem(sys.modules, "fiftyone", SimpleNamespace(
        dataset_exists=lambda name: False, Dataset=lambda **kwargs: dataset,
        Sample=lambda **kwargs: kwargs,
    ))
    paths = [(tmp_path / f"{i}.jpg", f"{i}.jpg") for i in range(3)]
    with pytest.raises(RuntimeError, match="write failed"):
        module.import_images(args(tmp_path, "--batch-size", "2", "--tags", "raw,raw,test"), paths)
    assert dataset.calls == 2
    assert dataset.samples[0]["tags"] == ["raw", "test"]
    assert dataset.samples[0]["relpath"] == "0.jpg"
    assert "persisted_samples=3" in caplog.text
