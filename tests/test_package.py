import importlib
import importlib.metadata
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

import ruts_adapt

ROOT = Path(__file__).parents[1]


def test_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", ruts_adapt.__version__)
    assert ruts_adapt.__version__ != "0.0.0"


def test_metadata():
    assert "русского текста" in ruts_adapt.__description__
    assert "__version__" in ruts_adapt.__all__
    assert ruts_adapt.__all__ == sorted(ruts_adapt.__all__)


def test_version_fallback(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    try:
        assert importlib.reload(ruts_adapt).__version__ == "0.0.0"
    finally:
        monkeypatch.undo()
        importlib.reload(ruts_adapt)
    assert ruts_adapt.__version__ != "0.0.0"


def test_entry_point():
    (point,) = importlib.metadata.entry_points(group="console_scripts", name="ruts-adapt")
    assert point.value == "ruts_adapt.cli:main"


def test_package_data():
    assert (Path(ruts_adapt.__file__).parent / "py.typed").is_file()


@pytest.mark.skipif(shutil.which("uv") is None, reason="колесо собирает uv")
def test_wheel_contents(tmp_path):
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT)],
        capture_output=True,
        check=True,
    )
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "ruts_adapt/py.typed" in names
    assert all(name.startswith(("ruts_adapt/", "ruts_adapt-")) for name in names)
