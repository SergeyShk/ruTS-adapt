from collections.abc import Mapping
from dataclasses import dataclass, field
from os import environ as os_environ
from typing import Literal, Self, get_args

ENV_PREFIX = "RUTS_ADAPT_"
Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORTS: tuple[Effort, ...] = get_args(Effort)
SWITCHES = {"1": True, "0": False}


@dataclass(frozen=True)
class ModelSettings:
    """
    Настройки вызова модели для одной роли

    Описание:
        Ответ модели приходит потоком, поэтому max_tokens может быть большим;
        в него входит и мышление модели, а не только ответ

    Атрибуты:
        model (str): Модель
        effort (str): Усилие модели (low, medium, high, xhigh, max)
        max_tokens (int): Наибольшее число токенов ответа вместе с мышлением
        fallbacks (bool): При отказе повторить запрос на резервной модели,
            которую API выбирает по категории отказа
    """

    model: str = "claude-opus-5-5"
    effort: Effort = "high"
    max_tokens: int = 64_000
    fallbacks: bool = True


@dataclass(frozen=True)
class Settings:
    """
    Настройки ruTS-adapt

    Описание:
        Писатель переписывает текст, судья выписывает факты исходника и проверяет
        по ним черновики. Настройка роли читается из переменной окружения
        RUTS_ADAPT_<РОЛЬ>_<НАСТРОЙКА> (RUTS_ADAPT_JUDGE_EFFORT), резервная модель
        для обеих ролей - из RUTS_ADAPT_FALLBACKS (1 или 0); если переменная
        не задана, остается значение по умолчанию

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
            >>> settings = Settings.from_env({"RUTS_ADAPT_JUDGE_EFFORT": "max"})
            >>> settings.writer.effort, settings.judge.effort, settings.judge.model
            ('high', 'max', 'claude-opus-5-5')
        """
        environ = os_environ if environ is None else environ
        fallbacks = _switch(environ, "FALLBACKS", ModelSettings.fallbacks)
        return cls(
            writer=_model_settings(environ, "WRITER", fallbacks),
            judge=_model_settings(environ, "JUDGE", fallbacks),
        )


def _model_settings(environ: Mapping[str, str], role: str, fallbacks: bool) -> ModelSettings:
    """Настройки роли из переменных RUTS_ADAPT_<РОЛЬ>_*"""
    model = environ.get(f"{ENV_PREFIX}{role}_MODEL", ModelSettings.model)
    if not model.strip():
        raise ValueError(f"{ENV_PREFIX}{role}_MODEL не должна быть пустой")
    return ModelSettings(
        model=model,
        effort=_effort(environ, f"{role}_EFFORT"),
        max_tokens=_positive_int(environ, f"{role}_MAX_TOKENS", ModelSettings.max_tokens),
        fallbacks=fallbacks,
    )


def _effort(environ: Mapping[str, str], name: str) -> Effort:
    """Усилие модели из переменной или значение по умолчанию"""
    variable = ENV_PREFIX + name
    raw = environ.get(variable, ModelSettings.effort)
    if raw not in EFFORTS:
        raise ValueError(
            f"{variable} должна быть одним из значений {', '.join(EFFORTS)}, получено {raw!r}"
        )
    return raw


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
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
