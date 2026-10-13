from pathlib import Path

import pytest
from platformdirs import user_data_dir

from ruts_adapt.settings import DEFAULT_DATA_DIR, EFFORTS, ModelSettings, Settings


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
    assert settings.writer == ModelSettings("google_genai:gemini-2.5-pro", "minimal", None, True)
    assert settings.judge == ModelSettings("openai:gpt-5", None, 16_000, True)


def test_default_model_tuning():
    """Усилие high и лимит 64 000 - для модели по умолчанию, даже заданной явно"""
    settings = Settings.from_env({"RUTS_ADAPT_WRITER_MODEL": "anthropic:claude-opus-5-5"})
    assert settings.writer == ModelSettings()
    other = Settings.from_env({"RUTS_ADAPT_WRITER_MODEL": "anthropic:claude-haiku-4-5"})
    assert other.writer == ModelSettings("anthropic:claude-haiku-4-5", None, None)


def test_ollama_model():
    """Имя локальной модели само содержит двоеточие: провайдер - до первого"""
    settings = Settings.from_env({"RUTS_ADAPT_WRITER_MODEL": "ollama:qwen3:30b"})
    assert (settings.writer.provider, settings.writer.name) == ("ollama", "qwen3:30b")
    assert (settings.writer.effort, settings.writer.max_tokens) == (None, None)


def test_ollama_context():
    """Окно контекста задается только локальной модели: умолчание сервера слишком мало"""
    assert Settings().writer.context is None
    settings = Settings.from_env(
        {
            "RUTS_ADAPT_WRITER_MODEL": "ollama:qwen3:30b",
            "RUTS_ADAPT_JUDGE_MODEL": "ollama:gpt-oss:20b",
            "RUTS_ADAPT_JUDGE_CONTEXT": "65536",
        }
    )
    assert (settings.writer.context, settings.judge.context) == (32_768, 65_536)


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


def test_strip():
    """Пробелы по краям значений не меняют смысла: модель по умолчанию остается ею"""
    settings = Settings.from_env(
        {
            "RUTS_ADAPT_WRITER_MODEL": "anthropic:claude-opus-5-5 ",
            "RUTS_ADAPT_JUDGE_EFFORT": " medium",
            "RUTS_ADAPT_FALLBACKS": "0\n",
        }
    )
    assert (settings.writer.effort, settings.writer.max_tokens) == ("high", 64_000)
    assert settings.judge.effort == "medium"
    assert settings.writer.fallbacks is False


def test_from_os_environ(monkeypatch):
    monkeypatch.setenv("RUTS_ADAPT_WRITER_EFFORT", "low")
    assert Settings.from_env().writer.effort == "low"


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        (
            {"RUTS_ADAPT_WRITER_MODEL": "claude-opus-5-5"},
            "RUTS_ADAPT_WRITER_MODEL должна иметь вид провайдер:модель с провайдером "
            "anthropic, openai, google_genai, ollama, получено 'claude-opus-5-5'",
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
            {"RUTS_ADAPT_WRITER_CONTEXT": "8192"},
            "RUTS_ADAPT_WRITER_CONTEXT задается только для ollama, модель 'anthropic:claude-opus-5-5'",
        ),
        (
            {"RUTS_ADAPT_JUDGE_MODEL": "ollama:qwen3:30b", "RUTS_ADAPT_JUDGE_CONTEXT": "32k"},
            "RUTS_ADAPT_JUDGE_CONTEXT должна быть целым положительным числом, получено '32k'",
        ),
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [("~/ruts-adapt", Path.home().resolve() / "ruts-adapt"), ("", DEFAULT_DATA_DIR)],
)
def test_data_dir(value, expected):
    """Каталог данных задает RUTS_ADAPT_DATA_DIR, пустая - не задает"""
    assert Settings.from_env({"RUTS_ADAPT_DATA_DIR": value}).data_dir == expected


def test_data_dir_default():
    assert Settings.from_env({}).data_dir == Path(user_data_dir("ruts-adapt", appauthor=False))
