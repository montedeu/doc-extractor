"""Нормализация текста документа для сверки с текстом (ADR 0001)."""

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from doc_extractor.numerals import find_amounts_in_words

_SPACES = "   "

_HOMOGLYPHS = str.maketrans(
    {
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
        "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
        "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к",
        "m": "м", "o": "о", "p": "р", "t": "т", "x": "х", "y": "у",
    }
)

_OCR_DIGITS = str.maketrans(
    {"O": "0", "О": "0", "o": "0", "о": "0", "l": "1", "I": "1", "І": "1"}
)

_OC = "0-9OОoоlIІ"
# Токен без пробелов из цифр и OCR-двойников (с точкой/запятой внутри).
_OCR_TOKEN = re.compile(rf"(?<!\w)[{_OC}][{_OC}.,]*(?!\w)")
# «25 OOO»: токен из трёх O/О сразу после цифры через один пробел, не перед словом.
_OCR_THOUSANDS = re.compile(
    rf"(?<=\d)([{_SPACES}])([OОoо]{{3}})(?!\w)(?![{_SPACES}]+[^\W\d_])"
)

_NUMBER = re.compile(
    rf"(?<!\d)\d+(?:[{_SPACES}]\d{{3}}(?!\d))*(?:[.,]\d{{3}}(?!\d))*(?:[.,]\d+)?"
)

_MONTH_STEMS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "мамыр": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11,
    "декабр": 12,
    "қаңтар": 1, "ақпан": 2, "наурыз": 3, "сәуір": 4,
    "маусым": 6, "шілде": 7, "тамыз": 8, "қыркүйек": 9, "қазан": 10,
    "қараша": 11, "желтоқсан": 12,
}
_MONTH_EXACT = {"мая": 5, "май": 5}
_WORD = r"[а-яёәіңғүұқөһ]+"
_DAY = r"[«\"(\[]?(\d{1,2})[»\")\]]?"
_DATE_NUMERIC = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4}|\d{2})(?!\d)")
_DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_DATE_WORDS = re.compile(rf"(?<!\d){_DAY}\s+({_WORD})\s+(\d{{4}})")
_DATE_KZ_ORDER = re.compile(
    rf"(?<!\d)(\d{{4}})\s*(?:жылғы|жылы|жыл|ж)\.?\s*{_DAY}\s+({_WORD})"
)


def _month(word: str) -> int | None:
    if word in _MONTH_EXACT:
        return _MONTH_EXACT[word]
    for stem, n in _MONTH_STEMS.items():
        if word.startswith(stem):
            return n
    return None


def fold_homoglyphs(text: str) -> str:
    """Латинские двойники кириллицы -> кириллица (для сравнения строк)."""
    return text.translate(_HOMOGLYPHS)


def fix_ocr_digits(text: str) -> str:
    """O/О/o -> 0, l/I -> 1 только внутри токена без пробелов с настоящей цифрой.

    Отдельные «ООО»/«OOO» не трогаются, кроме тройки O сразу после цифры через
    пробел («25 OOO»): она считается группой тысяч.
    """

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return tok.translate(_OCR_DIGITS) if any(c.isdigit() for c in tok) else tok

    text = _OCR_TOKEN.sub(repl, text)
    return _OCR_THOUSANDS.sub(lambda m: m.group(1) + "000", text)


def normalize_string(s: str) -> str:
    """Та же нормализация, что у DocFacts.folded, для строки из ответа модели."""
    s = fold_homoglyphs(fix_ocr_digits(s)).casefold()
    return " ".join(s.split())


def _quant(d: Decimal) -> Decimal:
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _variants(tok: str) -> set[str]:
    """Толкования одного токена без пробелов: десятичная запятая/точка, тысячи."""
    seps = re.findall(r"[.,]", tok)
    if not seps:
        return {tok}
    if "." in tok and "," in tok:
        dec = tok[max(tok.rfind("."), tok.rfind(",")) ]
        thousands = "," if dec == "." else "."
        return {tok.replace(thousands, "").replace(dec, ".")}
    if len(seps) > 1:
        return {re.sub(r"[.,]", "", tok)}
    out = {tok.replace(",", ".")}
    if len(tok) - tok.rfind(seps[0]) - 1 == 3:
        out.add(re.sub(r"[.,]", "", tok))
    return out


def _parse_numbers(text: str) -> set[Decimal]:
    """Склеенное число и его части («2 500» -> 2500, 2, 500): ложное отклонение хуже."""
    out: set[Decimal] = set()
    for m in _NUMBER.finditer(text):
        tok = m.group(0)
        parts = re.split(f"[{_SPACES}]", tok)
        # группа «000» не должна давать число 0
        cands = {v for p in parts if p.strip("0.,") for v in _variants(p)}
        if len(parts) > 1:
            cands |= _variants("".join(parts))
        for v in cands:
            try:
                out.add(_quant(Decimal(v)))
            except InvalidOperation:
                pass
    return out


def _parse_dates(text: str) -> set[date]:
    out: set[date] = set()

    def add(y: int, m: int | None, d: int) -> None:
        if m is None:
            return
        try:
            out.add(date(y, m, d))
        except ValueError:
            pass

    for d, m, y in _DATE_NUMERIC.findall(text):
        add(int(y) if len(y) == 4 else 2000 + int(y), int(m), int(d))
    for y, m, d in _DATE_ISO.findall(text):
        add(int(y), int(m), int(d))
    for d, word, y in _DATE_WORDS.findall(text):
        add(int(y), _month(word), int(d))
    for y, d, word in _DATE_KZ_ORDER.findall(text):
        add(int(y), _month(word), int(d))
    return out


@dataclass
class BinCandidate:
    digits: str  # 12 цифр после fix_ocr_digits
    raw: str  # как в тексте
    line: str  # строка документа целиком
    ocr_fixed: bool  # raw != digits
    start: int = 0  # позиция raw в строке
    prev_lines: list[str] = field(default_factory=list)  # до 5 строк абзаца выше, ближайшая первой


# 12 цифр/OCR-двойников; слева и справа любой символ, кроме буквы и цифры, или край строки.
# Десятичная часть («…012.50») кандидата не делает.
_BIN_TOKEN = re.compile(rf"(?<![^\W_])[{_OC}]{{12}}(?![^\W_])(?![.,]\d)")


def _bin_candidates(text: str) -> list[BinCandidate]:
    out: list[BinCandidate] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        prev: list[str] = []
        for x in reversed(lines[:idx]):
            if not x.strip() or len(prev) == 5:
                break
            prev.append(x)
        for m in _BIN_TOKEN.finditer(line):
            raw = m.group(0)
            digits = fix_ocr_digits(raw)
            if digits.isascii() and digits.isdigit():
                out.append(BinCandidate(digits, raw, line, raw != digits, m.start(), prev))
    return out


@dataclass
class DocFacts:
    text: str
    folded: str
    numbers: set[Decimal]
    dates: set[date]
    raw_numbers: set[Decimal]  # без исправления OCR: отличает ocr_corrected от ok
    amounts_in_words: list[int] = field(default_factory=list)
    bin_candidates: list[BinCandidate] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)  # нормализованные строки текста

    def has_number(self, x: float) -> bool:
        return _quant(Decimal(str(x))) in self.numbers

    def has_raw_number(self, x: float) -> bool:
        return _quant(Decimal(str(x))) in self.raw_numbers


def analyze(text: str) -> DocFacts:
    fixed = fix_ocr_digits(text)
    words = find_amounts_in_words(text)
    word_nums = {_quant(Decimal(n)) for n in words}
    return DocFacts(
        text=text,
        folded=normalize_string(text),
        numbers=_parse_numbers(fixed) | word_nums,
        dates=_parse_dates(normalize_string(text)),
        raw_numbers=_parse_numbers(text) | word_nums,
        amounts_in_words=words,
        bin_candidates=_bin_candidates(text),
        lines=[normalize_string(line) for line in text.splitlines()],
    )


_NAME_QUOTES = str.maketrans({c: " " for c in "«»\"“”„‟'"})
_ZERO_BETWEEN_LETTERS = re.compile(r"(?<=[^\W\d_])0(?=[^\W\d_])")
_NAME_PUNCT = re.compile(r"[,;:.]")
_SLASH = re.compile(r"\s*/\s*")
# после fold_homoglyphs + casefold: ТОО/TOO/T0O -> тоо, LLP -> llр, LLC -> llс
_OPF = re.compile(r"(?<!\w)(?:ооо|оао|зао|ао|ип|тоо|жшс|llр|llс)(?!\w)")


def _core(name: str, opf_sub: str) -> str:
    s = name.translate(_NAME_QUOTES)
    prev = None
    while prev != s:  # «T00»: lookaround не видит соседний 0, нужен повтор
        prev = s
        s = _ZERO_BETWEEN_LETTERS.sub("О", s)
    s = fold_homoglyphs(s).casefold().replace("ё", "е")
    s = _NAME_PUNCT.sub(" ", _OPF.sub(f" {opf_sub} ", s))
    return _SLASH.sub("/", " ".join(s.split()))


def name_core(name: str) -> str:
    """Название без ОПФ и кавычек, для сравнения; правила как у scoring.normalize_name."""
    return _core(name, "").strip()


def name_core_text(text: str) -> str:
    """То же для текста документа; ОПФ заменяется на «|», чтобы не склеивать соседей."""
    return _core(text, "|")


# Публичные имена для conflicts.py
NUMBER_RE = _NUMBER
SPACES_RE = _SPACES
number_variants = _variants
parse_dates = _parse_dates
