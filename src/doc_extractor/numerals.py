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
_FORMS = {
    "одну": 1, "одной": 1, "двух": 2, "трёх": 3, "трех": 3,
    "четырёх": 4, "четырех": 4,
    "двухсот": 200, "трёхсот": 300, "трехсот": 300, "четырёхсот": 400,
    "четырехсот": 400, "пятисот": 500, "шестисот": 600, "семисот": 700,
    "восьмисот": 800, "девятисот": 900,
}
_SMALL = {**_UNITS, **_TENS, **_HUNDREDS, **_FORMS}
_SCALES = {
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "тысячу": 1000, "мың": 1000,
    "миллион": 10**6, "миллиона": 10**6, "миллионов": 10**6,
}
_WORD = re.compile(r"[^\W\d_]+")
_JOINERS = " \t\n\r\u00a0\u202f\u2009-"
# маркеры валюты сверки (grounding._CURRENCY) плюс рубли; после fold_homoglyphs «EUR» -> «еur»
_CURRENCY = re.compile(r"(тенге|теңге|тг|рубл\w*|долл\w*|евро|usd|[eе]ur|кzт)$")
_CURRENCY_SIGNS = "₸€$"
# фраза из одного слова масштаба без числительного перед ним («тысяч тенге») — заголовок, не сумма
_BARE_SCALES = {"тысяч", "тысячи", "мың", "миллиона", "миллионов"}


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


def _is_currency(word: str) -> bool:
    from doc_extractor.normalize import fold_homoglyphs

    return bool(_CURRENCY.match(fold_homoglyphs(word.casefold())))


def find_amounts_in_words(text: str) -> list[int]:
    """Суммы прописью: последовательности числительных, за которыми идёт валюта.

    Перед фразой в той же строке не должно быть цифры («12 тысяч рублей» — не сумма).
    """
    from doc_extractor.normalize import fold_homoglyphs

    text = fold_homoglyphs(text)
    toks = [(m.start(), m.end(), m.group(0).casefold()) for m in _WORD.finditer(text)]
    out: list[int] = []
    i = 0
    while i < len(toks):
        if not _is_numeral(toks[i][2]):
            i += 1
            continue
        j = i + 1
        while (
            j < len(toks)
            and _is_numeral(toks[j][2])
            and not text[toks[j - 1][1] : toks[j][0]].strip(_JOINERS)
        ):
            j += 1
        before = text[:toks[i][0]].rstrip(" \t\u00a0\u202f\u2009")
        after_ok = (
            j < len(toks)
            and not text[toks[j - 1][1] : toks[j][0]].strip(_JOINERS)
            and _is_currency(toks[j][2])
        )
        if not after_ok:
            sign = text[toks[j - 1][1] :].lstrip(" \t\u00a0\u202f\u2009")[:1]
            after_ok = sign != "" and sign in _CURRENCY_SIGNS
        bare = j - i == 1 and toks[i][2] in _BARE_SCALES
        if after_ok and not bare and not (before and before[-1].isdigit()):
            out.append(_value([t[2] for t in toks[i:j]]))
        i = j
    return out
