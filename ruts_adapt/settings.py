from collections.abc import Mapping
from dataclasses import dataclass, field
from os import environ as os_environ
from typing import Self

ENV_PREFIX = "RUTS_ADAPT_"
# Уровни усилия, которые принимает модель чата LangChain каждого провайдера (reasoning_effort)
EFFORTS = {
    "anthropic": ("low", "medium", "high", "xhigh", "max"),
    "openai": ("none", "minimal", "low", "medium", "high", "xhigh"),
    "google_genai": ("minimal", "low", "medium", "high"),
}
SWITCHES = {"1": True, "0": False}
DEFAULT_MODEL = "anthropic:claude-opus-5-5"


@dataclass(frozen=True)
class ModelSettings:
    """
    Настройки вызова модели для одной роли

    Описание:
        Модель задается как провайдер:модель, как в init_chat_model LangChain;
        провайдеры - anthropic, openai, google_genai (Gemini). Усилие - уровень
        рассуждения модели, у каждого провайдера свой набор. В max_tokens входит
        и мышление модели, а не только ответ. None - умолчание провайдера:
        усилие high и лимит 64 000 подобраны для модели по умолчанию

    Атрибуты:
        model (str): Провайдер и модель
        effort (str | None): Усилие модели
        max_tokens (int | None): Наибольшее число токенов ответа вместе с мышлением
        fallbacks (bool): Только для anthropic: при отказе повторить запрос
            на резервной модели, которую API выбирает по категории отказа
    """

    model: str = DEFAULT_MODEL
    effort: str | None = "high"
    max_tokens: int | None = 64_000
    fallbacks: bool = True

    @property
    def provider(self) -> str:
        """Провайдер модели"""
        return self.model.partition(":")[0]

    @property
    def name(self) -> str:
        """Модель без провайдера"""
        return self.model.partition(":")[2]


@dataclass(frozen=True)
class Settings:
    """
    Настройки ruTS-adapt

    Описание:
        Писатель переписывает текст, судья выписывает факты исходника и проверяет
        по ним черновики. Настройка роли читается из переменной окружения
        RUTS_ADAPT_<РОЛЬ>_<НАСТРОЙКА> (RUTS_ADAPT_JUDGE_EFFORT), резервная модель
        для обеих ролей - из RUTS_ADAPT_FALLBACKS (1 или 0); если переменная
        не задана, остается значение по умолчанию. Для модели не по умолчанию
        усилие и лимит по умолчанию - умолчания провайдера, как и пустое усилие

    Атрибуты:
        writer (ModelSettings): Настройки писателя
        judge (ModelSettings): Настройки судьи
    """

    writer: ModelSettings = field(default_factory=ModelSettings)
    judge: ModelSettings = field(default_factory=ModelSettings)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        """
        Чтение настроек из окружения

        Аргументы:
            environ (Mapping[str, str]): Переменные окружения; None - os.environ

        Вывод:
            Settings: Настройки

        Исключения:
            ValueError: Если значение переменной недопустимо

        Пример использования:
            >>> settings = Settings.from_env(
            ...     {"RUTS_ADAPT_JUDGE_MODEL": "openai:gpt-5", "RUTS_ADAPT_JUDGE_EFFORT": "medium"}
            ... )
            >>> settings.writer.model, settings.judge.provider, settings.judge.effort
            ('anthropic:claude-opus-5-5', 'openai', 'medium')
            >>> settings.writer.max_tokens, settings.judge.max_tokens
            (64000, None)
        """
        environ = os_environ if environ is None else environ
        fallbacks = _switch(environ, "FALLBACKS", ModelSettings.fallbacks)
        return cls(
            writer=_model_settings(environ, "WRITER", fallbacks),
            judge=_model_settings(environ, "JUDGE", fallbacks),
        )


def _model_settings(environ: Mapping[str, str], role: str, fallbacks: bool) -> ModelSettings:
    """Настройки роли из переменных RUTS_ADAPT_<РОЛЬ>_*"""
    variable = f"{ENV_PREFIX}{role}_MODEL"
    model = environ.get(variable, DEFAULT_MODEL)
    # Умолчания усилия и лимита подобраны для модели по умолчанию, а не для любой модели провайдера
    defaults = ModelSettings() if model == DEFAULT_MODEL else ModelSettings(model, None, None)
    provider, _, name = model.partition(":")
    if provider not in EFFORTS or not name:
        raise ValueError(
            f"{variable} должна иметь вид провайдер:модель с провайдером "
            f"{', '.join(EFFORTS)}, получено {model!r}"
        )
    variable = f"{ENV_PREFIX}{role}_EFFORT"
    effort = environ.get(variable, defaults.effort) or None
    if effort is not None and effort not in EFFORTS[provider]:
        raise ValueError(
            f"{variable} для {provider} должна быть одним из значений "
            f"{', '.join(EFFORTS[provider])} или пустой, получено {effort!r}"
        )
    return ModelSettings(
        model=model,
        effort=effort,
        max_tokens=_positive_int(environ, f"{role}_MAX_TOKENS", defaults.max_tokens),
        fallbacks=fallbacks,
    )


def _positive_int(environ: Mapping[str, str], name: str, default: int | None) -> int | None:
    """Целое положительное значение переменной или значение по умолчанию"""
    variable = ENV_PREFIX + name
    if variable not in environ:
        return default
    raw = environ[variable]
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{variable} должна быть целым положительным числом, получено {raw!r}")
    return value


def _switch(environ: Mapping[str, str], name: str, default: bool) -> bool:
    """Переключатель 1 или 0 из переменной или значение по умолчанию"""
    variable = ENV_PREFIX + name
    if variable not in environ:
        return default
    raw = environ[variable]
    if raw not in SWITCHES:
        raise ValueError(f"{variable} должна быть 1 или 0, получено {raw!r}")
    return SWITCHES[raw]
