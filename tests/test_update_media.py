"""Tests for update_media hash failure handling."""
import importlib.util
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parents[1]


def load_script():
    path = ROOT / "scripts" / "update_media.py"
    spec = importlib.util.spec_from_file_location("update_media", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


media = load_script()


class FakeSample(dict):
    id = "abc"

    def has_field(self, name):
        return name in self

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)


def test_phash_truncated_image_is_skipped(tmp_path):
    good = tmp_path / "good.jpg"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(good)
    bad = tmp_path / "truncated.jpg"
    data = good.read_bytes()
    bad.write_bytes(data[: max(40, len(data) // 4)])

    sample = FakeSample()
    written, failures = media.enrich_sample_hashes(
        sample, bad, ["sha256", "phash"], overwrite=False
    )
    assert failures == 1
    assert written == ["sha256"]
    assert sample["sha256"]
    assert "phash" not in sample
