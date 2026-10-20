import math

import pytest
from ruts import BasicStats, DiversityStats, ReadabilityStats, SourceError
from ruts.readability_stats import grade_to_age, level_norms

from ruts_adapt.measure import MAX_GRADE, MIN_GRADE, Direction, Target, measure

TALE = "Жили-были дед да баба. Была у них курочка Ряба. Снесла курочка яичко. Яичко не простое, золотое."
DECREE = (
    "Настоящее постановление устанавливает порядок предоставления субсидий юридическим лицам, "
    "индивидуальным предпринимателям и физическим лицам - производителям товаров, работ и услуг "
    "в целях возмещения недополученных доходов и финансового обеспечения затрат в связи "
    "с осуществлением деятельности, предусмотренной муниципальными программами."
)


def test_measure():
    m = measure(DECREE)
    bs = BasicStats(DECREE)
    rs = ReadabilityStats(DECREE)
    assert m.grade == rs.consensus_grade
    assert m.level == rs.describe_grade()
    assert (m.n_words, m.n_sents) == (bs.n_words, bs.n_sents) == (35, 1)
    assert m.words_per_sent == 35.0
    assert m.syllables_per_word == bs.n_syllables / bs.n_words
    assert m.letters_per_word == sum(n * count for n, count in bs.c_letters.items()) / bs.n_words
    assert m.complex_words == bs.count_words_by_syllables(rs.smog_complex_syl_factor) / bs.n_words
    assert rs.smog_complex_syl_factor == 5
    assert m.mtld == DiversityStats(DECREE).mtld


def test_measure_order():
    """Сказка читается проще постановления: ниже класс, короче предложения и слова, меньше сложных"""
    tale, decree = measure(TALE), measure(DECREE)
    assert tale.level == "1-3-й класс (6-8 лет)"
    assert decree.level == "аспирантура (старше 22 лет)"
    assert tale.grade < decree.grade
    assert tale.words_per_sent < decree.words_per_sent
    assert tale.syllables_per_word < decree.syllables_per_word
    assert tale.letters_per_word < decree.letters_per_word
    assert tale.complex_words < decree.complex_words


def test_measure_mtld_no_repeats():
    """MTLD текста без повторяющихся слов не определен: ruts дает inf, с одним повтором - число"""
    assert measure("Кот спит. Кот ест.").mtld == pytest.approx(4.48)
    assert measure("Утром рыжий котенок выбрался из корзинки и побежал во двор").mtld == math.inf


@pytest.mark.parametrize("text", ["", "  ", "..."])
def test_measure_no_words(text):
    with pytest.raises(SourceError):
        measure(text)


@pytest.mark.parametrize("grade", range(MIN_GRADE, MAX_GRADE + 1))
def test_target_from_grade(grade):
    target = Target.from_grade(grade)
    assert target.grade == grade
    assert target.low <= grade <= target.high
    assert target.level == grade_to_age(grade) == grade_to_age(target.low)
    assert target.gap(target.low) == target.gap(target.high) == 0


@pytest.mark.parametrize(
    ("grade", "low", "high", "level"),
    [
        (1, 1, 3, "1-3-й класс (6-8 лет)"),
        (5, 4, 6, "4-6-й класс (9-11 лет)"),
        (11, 10, 11, "10-11-й класс (15-16 лет)"),
        (17, 15, 17, "4-6-й курс вуза (20-22 года)"),
    ],
)
def test_target_levels(grade, low, high, level):
    assert Target.from_grade(grade) == Target(grade, low, high, level)


@pytest.mark.parametrize("grade", [0, 18, -1, 5.0, True, "5", None])
def test_target_invalid(grade):
    with pytest.raises(ValueError, match=f"целым числом от 1 до 17, получено {grade!r}$"):
        Target.from_grade(grade)


@pytest.mark.parametrize(
    ("target", "grade", "gap"),
    [
        (5, 6.0, 0.0),
        (5, 6.4, 0.0),
        (5, 6.5, 0.5),
        (5, 8.5, 2.5),
        (5, 3.5, 0.0),
        (5, 3.4, -0.6),
        (5, -4.0, -8.0),
        (2, 0.4, 0.0),
        (2, -4.0, 0.0),
        (17, 17.4, 0.0),
        (17, 22.0, 5.0),
        (14, 15.0, 1.0),
    ],
)
def test_gap(target, grade, gap):
    """Граница ступени - по округлению ruTS: 6.4 - еще 6-й класс, 6.5 - уже 7-й"""
    assert Target.from_grade(target).gap(grade) == pytest.approx(gap)


@pytest.mark.parametrize(
    ("grade", "direction"),
    [(7.0, Direction.SIMPLIFY), (5.0, Direction.KEEP), (3.0, Direction.COMPLICATE)],
)
def test_direction(grade, direction):
    assert Target.from_grade(5).direction(grade) is direction


def test_direction_texts():
    target = Target.from_grade(5)
    assert target.direction(measure(DECREE).grade) is Direction.SIMPLIFY
    assert target.direction(measure(TALE).grade) is Direction.COMPLICATE


@pytest.mark.parametrize(("grade", "norm"), [(2, True), (6, True), (8, False), (11, False)])
def test_target_norm(grade, norm):
    """Нормы ruTS - только у 1-3 и 4-6-го класса"""
    target = Target.from_grade(grade)
    assert target.norm == (level_norms(grade) if norm else None)
    assert (target.describe_norm() is not None) is norm


def test_target_norm_ignored():
    """Нормы 10-11-го класса в ruTS есть, но граф их не берет"""
    assert level_norms(11) is not None
    assert Target.from_grade(11).norm is None
