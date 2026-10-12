import pytest

from ruts_adapt.settings import EFFORTS, ModelSettings, Settings


def test_defaults():
    assert Settings.from_env({}) == Settings()
    for role in (Settings().writer, Settings().judge):
        assert role == ModelSettings("claude-opus-5-5", "high", 64_000, True)


def test_roles():
    settings = Settings.from_env(
        {
            "RUTS_ADAPT_WRITER_MODEL": "claude-sonnet-5-5",
            "RUTS_ADAPT_WRITER_EFFORT": "medium",
            "RUTS_ADAPT_JUDGE_MAX_TOKENS": "16000",
        }
    )
    assert settings.writer == ModelSettings("claude-sonnet-5-5", "medium", 64_000, True)
    assert settings.judge == ModelSettings("claude-opus-5-5", "high", 16_000, True)


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_fallbacks(value, expected):
    """Резервная модель включается и выключается для обеих ролей сразу"""
    settings = Settings.from_env({"RUTS_ADAPT_FALLBACKS": value})
    assert settings.writer.fallbacks is settings.judge.fallbacks is expected


@pytest.mark.parametrize("effort", EFFORTS)
def test_efforts(effort):
    assert Settings.from_env({"RUTS_ADAPT_JUDGE_EFFORT": effort}).judge.effort == effort


def test_from_os_environ(monkeypatch):
    monkeypatch.setenv("RUTS_ADAPT_WRITER_EFFORT", "low")
    assert Settings.from_env().writer.effort == "low"


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("RUTS_ADAPT_WRITER_EFFORT", "HIGH", "одним из значений low, medium, high, xhigh, max"),
        ("RUTS_ADAPT_JUDGE_EFFORT", "", "одним из значений"),
        ("RUTS_ADAPT_WRITER_MAX_TOKENS", "0", "целым положительным числом, получено '0'"),
        ("RUTS_ADAPT_JUDGE_MAX_TOKENS", "16k", "целым положительным числом"),
        ("RUTS_ADAPT_JUDGE_MODEL", " ", "не должна быть пустой"),
        ("RUTS_ADAPT_FALLBACKS", "yes", "должна быть 1 или 0, получено 'yes'"),
    ],
)
def test_invalid(variable, value, message):
    with pytest.raises(ValueError, match=message) as info:
        Settings.from_env({variable: value})
    assert str(info.value).startswith(variable)
