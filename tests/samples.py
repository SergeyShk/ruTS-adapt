from ruts_adapt.tasks import FactCheck, FactStatus, Verdict

DECREE = (
    "Настоящее постановление устанавливает порядок предоставления субсидий юридическим лицам "
    "в целях возмещения недополученных доходов в связи с осуществлением деятельности."
)
MEDIUM = (
    "Постановление устанавливает порядок предоставления субсидий организациям, "
    "чтобы возместить недополученные доходы от их деятельности."
)
SIMPLE = "Есть новые правила. Фирмам дают деньги. Деньги платят, если фирма не получила доход."
FACTS = [
    "Постановление устанавливает порядок субсидий",
    "Субсидии возмещают недополученные доходы",
]


def verdict(*statuses: FactStatus) -> Verdict:
    """Вердикт судьи по фактам FACTS в порядке номеров"""
    return Verdict(
        checks=[
            FactCheck(
                fact=number, status=status, quote="" if status is FactStatus.LOST else "цитата"
            )
            for number, status in enumerate(statuses, 1)
        ]
    )


KEPT = verdict(FactStatus.KEPT, FactStatus.KEPT)
ONE_LOST = verdict(FactStatus.KEPT, FactStatus.LOST)
