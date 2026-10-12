from collections import Counter
from enum import StrEnum

from anthropic import Anthropic
from pydantic import BaseModel, Field

from .measure import Direction, Measurement, Target
from .model import ModelError, Reply, ask
from .settings import ModelSettings

FACTS_SYSTEM = """\
Ты выписываешь факты из текста, чтобы потом проверить, сохранил ли их пересказ этого текста.

Факт - одно утверждение текста, которое можно найти или не найти в пересказе: кто что сделал, \
каким был, где, когда, сколько, почему. Каждый факт - короткое самостоятельное предложение \
на русском языке, понятное без соседних фактов: вместо местоимений - имена и названия. \
Выписывай только то, что сказано в тексте, без выводов и оценок. Порядок фактов - как в тексте."""

REWRITE_SYSTEM = """\
Ты адаптируешь русский текст для читателя заданного уровня: упрощаешь или усложняешь его, \
сохраняя содержание.

Уровень текста измеряется формулами удобочитаемости по длине предложений в словах, длине слов \
в слогах и буквах и доле слов от пяти слогов. Чтобы упростить текст, дели длинные предложения, \
заменяй длинные и редкие слова короткими и частыми. Чтобы усложнить, объединяй короткие \
предложения в сложные, используй книжную лексику и более точные, длинные слова.

Сохрани каждый факт из списка: его участников, числа, места, время и причины. Не добавляй \
фактов, которых нет в исходном тексте. Пиши связным литературным русским языком того же жанра, \
что и исходный текст. В ответе - только адаптированный текст, без заголовков и пояснений.

Если в сообщении есть прошлая попытка и критика к ней, эта попытка не прошла проверку уровня \
или фактов. Перепиши заново исходный текст, учитывая критику, а не правь прошлую попытку."""

JUDGE_SYSTEM = """\
Ты проверяешь, сохранил ли пересказ факты исходного текста.

Для каждого факта из списка реши, как он передан в пересказе:
- сохранен - факт есть в пересказе, пусть другими словами;
- искажен - факт передан с ошибкой: другой участник, число, место, время или причина;
- потерян - факта в пересказе нет.

Для сохраненного и искаженного факта приведи дословную цитату из пересказа, где он передан, \
для потерянного - пустую строку. Оцени каждый факт ровно один раз, указав его номер."""

DIRECTIONS = {Direction.SIMPLIFY: "Упрости", Direction.COMPLICATE: "Усложни"}


class Facts(BaseModel):
    """Факты исходного текста"""

    facts: list[str] = Field(description="Факты в порядке текста")


class Draft(BaseModel):
    """Адаптированный текст"""

    text: str = Field(description="Адаптированный текст целиком")


class FactStatus(StrEnum):
    """Как факт передан в черновике"""

    KEPT = "сохранен"
    DISTORTED = "искажен"
    LOST = "потерян"


class FactCheck(BaseModel):
    """Проверка одного факта"""

    fact: int = Field(description="Номер факта в списке")
    status: FactStatus
    quote: str = Field(description="Цитата из пересказа; для потерянного факта - пустая строка")


class Verdict(BaseModel):
    """Проверка всех фактов"""

    checks: list[FactCheck]


def extract_facts(client: Anthropic, settings: ModelSettings, text: str) -> Reply[list[str]]:
    """
    Извлечение фактов исходного текста

    Аргументы:
        client (Anthropic): Клиент API
        settings (ModelSettings): Настройки судьи
        text (str): Исходный текст

    Вывод:
        Reply: Факты в порядке текста

    Исключения:
        ModelError: Если ответ модели нельзя использовать или в нем нет фактов
    """
    reply = ask(client, settings, FACTS_SYSTEM, _tagged("текст", text), Facts)
    facts = [fact.strip() for fact in reply.output.facts if fact.strip()]
    if not facts:
        raise ModelError("Модель не нашла в тексте ни одного факта")
    return reply.with_output(facts)


def rewrite(
    client: Anthropic,
    settings: ModelSettings,
    source: str,
    measurement: Measurement,
    target: Target,
    facts: list[str],
    previous: str | None = None,
    critique: str | None = None,
) -> Reply[str]:
    """
    Адаптация исходного текста к ступени цели

    Описание:
        Каждая попытка переписывает исходный текст, а не прошлый черновик;
        прошлый черновик и критика к нему подсказывают, что исправить

    Аргументы:
        client (Anthropic): Клиент API
        settings (ModelSettings): Настройки писателя
        source (str): Исходный текст
        measurement (Measurement): Уровень и признаки исходного текста
        target (Target): Цель адаптации
        facts (list[str]): Факты, которые нужно сохранить
        previous (str): Прошлый черновик
        critique (str): Критика прошлого черновика

    Вывод:
        Reply: Адаптированный текст

    Исключения:
        ValueError: Если исходный текст уже на уровне цели
        ModelError: Если ответ модели нельзя использовать или текст пуст
    """
    direction = target.direction(measurement.grade)
    if direction is Direction.KEEP:
        raise ValueError(f"Текст уже на уровне цели: {target.level}")
    parts = [
        f"{DIRECTIONS[direction]} исходный текст для читателя уровня «{target.level}». "
        f"Сейчас он читается как «{measurement.level}»: в среднем "
        f"{measurement.words_per_sent:.1f} слова в предложении, "
        f"{measurement.syllables_per_word:.2f} слога и {measurement.letters_per_word:.2f} буквы "
        f"в слове, {measurement.complex_words:.0%} слов от пяти слогов.",
        _tagged("исходный_текст", source),
        _tagged("факты", _numbered(facts)),
    ]
    if previous is not None:
        parts.append(_tagged("прошлая_попытка", previous))
    if critique is not None:
        parts.append(_tagged("критика", critique))
    reply = ask(client, settings, REWRITE_SYSTEM, "\n\n".join(parts), Draft)
    text = reply.output.text.strip()
    if not text:
        raise ModelError("Модель вернула пустой текст")
    return reply.with_output(text)


def judge(
    client: Anthropic, settings: ModelSettings, facts: list[str], draft: str
) -> Reply[list[FactCheck]]:
    """
    Проверка фактов исходного текста в черновике

    Аргументы:
        client (Anthropic): Клиент API
        settings (ModelSettings): Настройки судьи
        facts (list[str]): Факты исходного текста
        draft (str): Черновик

    Вывод:
        Reply: Проверки фактов в порядке списка, номера с 1

    Исключения:
        ModelError: Если ответ модели нельзя использовать или судья оценил
            не каждый факт ровно один раз
    """
    prompt = "\n\n".join([_tagged("факты", _numbered(facts)), _tagged("пересказ", draft)])
    reply = ask(client, settings, JUDGE_SYSTEM, prompt, Verdict)
    counts = Counter(check.fact for check in reply.output.checks)
    expected = range(1, len(facts) + 1)
    missing = [number for number in expected if number not in counts]
    repeated = sorted(number for number, count in counts.items() if count > 1)
    unknown = sorted(number for number in counts if number not in expected)
    if missing or repeated or unknown:
        problems = [
            f"{label}: {', '.join(map(str, numbers))}"
            for label, numbers in (
                ("нет оценки", missing),
                ("несколько оценок", repeated),
                ("нет такого факта", unknown),
            )
            if numbers
        ]
        raise ModelError(f"Судья оценил не каждый факт ровно один раз ({'; '.join(problems)})")
    checks = sorted(reply.output.checks, key=lambda check: check.fact)
    return reply.with_output(checks)


def _tagged(tag: str, content: str) -> str:
    """Содержимое в тегах, чтобы модель не путала его с указаниями"""
    return f"<{tag}>\n{content}\n</{tag}>"


def _numbered(items: list[str]) -> str:
    """Нумерованный список с 1"""
    return "\n".join(f"{number}. {item}" for number, item in enumerate(items, 1))
