"""Суммы прописью (ru, kk) -> целые числа. Казахские формы с окончаниями не поддерживаются."""

import re

_UNITS: dict[str, int] = {}
for _i, _words in enumerate(
    [
        "ноль нөл", "один одна одно бір", "два две екі", "три үш", "четыре төрт",
        "пять бес", "шесть алты", "семь жеті", "восемь сегіз", "девять тоғыз",
        "десять он", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
        "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
    ]
):
    for _w in _words.split():
        _UNITS[_w] = _i

_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "жиырма": 20, "отыз": 30, "қырық": 40, "елу": 50, "алпыс": 60,
    "жетпіс": 70, "сексен": 80, "тоқсан": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800,
    "девятьсот": 900, "жүз": 100,
}
_SMALL = {**_UNITS, **_TENS, **_HUNDREDS}
_SCALES = {
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "мың": 1000,
    "миллион": 10**6, "миллиона": 10**6, "миллионов": 10**6,
}
_WORD = re.compile(r"[^\W\d_]+")
_JOINERS = " \t   -"


def _is_numeral(w: str) -> bool:
    return w in _SMALL or w in _SCALES


def _value(words: list[str]) -> int:
    total = cur = 0
    prev = 0  # предыдущее слово-единица: казахское «үш жүз» = 3 * 100
    for w in words:
        if w == "жүз" and 1 <= prev <= 9:
            cur += prev * 99
        elif w in _SCALES:
            total += (cur or 1) * _SCALES[w]
            cur = 0
        else:
            cur += _SMALL[w]
        prev = _SMALL.get(w, 0) if w in _UNITS else 0
    return total + cur


def parse_amount_words(phrase: str) -> int | None:
    """«үш жүз алпыс мың» -> 360000; не число -> None."""
    words = [w.casefold() for w in _WORD.findall(phrase)]
    if not words or not all(_is_numeral(w) for w in words):
        return None
    return _value(words)


def find_amounts_in_words(text: str) -> list[int]:
    """Все суммы прописью: максимальные последовательности слов-числительных."""
    out: list[int] = []
    run: list[str] = []
    pos = 0
    for m in _WORD.finditer(text):
        w = m.group(0).casefold()
        # слова одной фразы разделены только пробелами или дефисом
        if run and text[pos : m.start()].strip(_JOINERS):
            out.append(_value(run))
            run = []
        if _is_numeral(w):
            run.append(w)
        elif run:
            out.append(_value(run))
            run = []
        pos = m.end()
    if run:
        out.append(_value(run))
    return out
