import pytest


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    """Каталог данных теста вместо каталога пользователя: потоки адаптации пишутся сюда"""
    path = tmp_path / "data"
    monkeypatch.setenv("RUTS_ADAPT_DATA_DIR", str(path))
    return path
