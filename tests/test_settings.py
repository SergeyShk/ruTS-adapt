import pytest

from ruts_adapt.settings import EFFORTS, ModelSettings, Settings


def test_defaults():
    assert Settings.from_env({}) == Settings()
    for role in (Settings().writer, Settings().judge):
        assert role == ModelSettings("anthropic:claude-opus-5-5", "high", 64_000, True)
        assert (role.provider, role.name) == ("anthropic", "claude-opus-5-5")


def test_roles():
    settings = Settings.from_env(
        {
            "RUTS_ADAPT_WRITER_MODEL": "google_genai:gemini-2.5-pro",
            "RUTS_ADAPT_WRITER_EFFORT": "minimal",
            "RUTS_ADAPT_JUDGE_MODEL": "openai:gpt-5",
            "RUTS_ADAPT_JUDGE_MAX_TOKENS": "16000",
        }
    )
    assert settings.writer == ModelSettings("google_genai:gemini-2.5-pro", "minimal", 64_000, True)
    assert settings.judge == ModelSettings("openai:gpt-5", "high", 16_000, True)


def test_effort_default():
    """Пустое усилие - умолчание провайдера, для моделей без рассуждения"""
    settings = Settings.from_env(
        {"RUTS_ADAPT_JUDGE_MODEL": "openai:gpt-4.1", "RUTS_ADAPT_JUDGE_EFFORT": ""}
    )
    assert settings.judge.effort is None


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_fallbacks(value, expected):
    """Резервная модель включается и выключается для обеих ролей сразу"""
    settings = Settings.from_env({"RUTS_ADAPT_FALLBACKS": value})
    assert settings.writer.fallbacks is settings.judge.fallbacks is expected


@pytest.mark.parametrize(
    ("provider", "effort"),
    [(provider, effort) for provider, efforts in EFFORTS.items() for effort in efforts],
)
def test_efforts(provider, effort):
    environ = {"RUTS_ADAPT_WRITER_MODEL": f"{provider}:model", "RUTS_ADAPT_WRITER_EFFORT": effort}
    assert Settings.from_env(environ).writer.effort == effort


def test_from_os_environ(monkeypatch):
    monkeypatch.setenv("RUTS_ADAPT_WRITER_EFFORT", "low")
    assert Settings.from_env().writer.effort == "low"


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        (
            {"RUTS_ADAPT_WRITER_MODEL": "claude-opus-5-5"},
            "RUTS_ADAPT_WRITER_MODEL должна иметь вид провайдер:модель с провайдером "
            "anthropic, openai, google_genai, получено 'claude-opus-5-5'",
        ),
        (
            {"RUTS_ADAPT_JUDGE_MODEL": "gemini:gemini-2.5-pro"},
            "RUTS_ADAPT_JUDGE_MODEL должна иметь вид провайдер:модель",
        ),
        ({"RUTS_ADAPT_JUDGE_MODEL": "openai:"}, "RUTS_ADAPT_JUDGE_MODEL должна иметь вид"),
        (
            {"RUTS_ADAPT_WRITER_EFFORT": "HIGH"},
            "RUTS_ADAPT_WRITER_EFFORT для anthropic должна быть одним из значений "
            "low, medium, high, xhigh, max или пустой, получено 'HIGH'",
        ),
        (
            {
                "RUTS_ADAPT_JUDGE_MODEL": "google_genai:gemini-2.5-pro",
                "RUTS_ADAPT_JUDGE_EFFORT": "max",
            },
            "RUTS_ADAPT_JUDGE_EFFORT для google_genai должна быть одним из значений "
            "minimal, low, medium, high или пустой, получено 'max'",
        ),
        (
            {"RUTS_ADAPT_WRITER_MAX_TOKENS": "0"},
            "RUTS_ADAPT_WRITER_MAX_TOKENS должна быть целым положительным числом, получено '0'",
        ),
        ({"RUTS_ADAPT_JUDGE_MAX_TOKENS": "16k"}, "RUTS_ADAPT_JUDGE_MAX_TOKENS должна быть целым"),
        (
            {"RUTS_ADAPT_FALLBACKS": "yes"},
            "RUTS_ADAPT_FALLBACKS должна быть 1 или 0, получено 'yes'",
        ),
    ],
)
def test_invalid(environ, message):
    with pytest.raises(ValueError) as info:
        Settings.from_env(environ)
    assert str(info.value).startswith(message)
