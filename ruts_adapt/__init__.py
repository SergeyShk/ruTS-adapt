# Адаптация русского текста под уровень читателя (ruTS-adapt)
#
# Copyright (C) 2026
# Автор: Шкарин Сергей <kouki.sergey@gmail.com>
# URL: <https://github.com/SergeyShk/ruTS-adapt>

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ruts-adapt")
except PackageNotFoundError:
    __version__ = "0.0.0"
__description__ = """Адаптация русского текста под уровень читателя: граф LangGraph, в котором
текст переписывает модель, уровень измеряет ruTS, а сохранность смысла проверяет LLM-судья.
Требует версию Python 3.11 и выше"""
__author__ = "Шкарин Сергей"
__author_email__ = "kouki.sergey@gmail.com"

__all__ = ["__version__"]
