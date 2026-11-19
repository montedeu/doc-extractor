"""Нормализация текста документа для сверки с текстом (ADR 0001)."""

import re
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

_SPACES = "   "

_HOMOGLYPHS = str.maketrans(
    {
        "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
        "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
        "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х",
        "y": "у",
    }
)

_OCR_DIGITS = str.maketrans(
    {"O": "0", "О": "0", "o": "0", "о": "0", "l": "1", "I": "1", "І": "1"}
)

_OC = "0-9OОoоlIІ"
# Числовой фрагмент: цифры и OCR-двойники, группы по 3 через пробел, дробная часть.
# Не граничит с другими буквами (R410, 3x6, 80г не трогаем).
_OCR_FRAGMENT = re.compile(
    rf"(?<![^\W\d_])(?<!\d)"
    rf"[{_OC}]+(?:[{_SPACES}][{_OC}]{{3}}(?![{_OC}]))*"
    rf"(?:[.,][{_OC}]+)?"
    rf"(?![^\W\d_])"
)

_NUMBER = re.compile(rf"(?<![\d.,])\d+(?:[{_SPACES}]\d{{3}}(?!\d))*(?:[.,]\d+)?")

_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11,
    "декабр": 12,
    "қаңтар": 1, "ақпан": 2, "наурыз": 3, "сәуір": 4, "мамыр": 5,
    "маусым": 6, "шілде": 7, "тамыз": 8, "қыркүйек": 9, "қазан": 10,
    "қараша": 11, "желтоқсан": 12,
}
_MONTH_WORDS = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_NUMERIC = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")
_DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_DATE_WORDS = re.compile(
    rf"(?<!\d)(\d{{1,2}})\s+({_MONTH_WORDS})[а-яёәіңғүұқөһ]*\s+(\d{{4}})"
)


def fold_homoglyphs(text: str) -> str:
    """Латинские двойники кириллицы -> кириллица (для сравнения строк)."""
    return text.translate(_HOMOGLYPHS)


def fix_ocr_digits(text: str) -> str:
    """O/О/o -> 0, l/I -> 1 только в числовых фрагментах (есть хотя бы одна цифра)."""

    def repl(m: re.Match) -> str:
        frag = m.group(0)
        if not any(c.isdigit() for c in frag):
            return frag
        return frag.translate(_OCR_DIGITS)

    return _OCR_FRAGMENT.sub(repl, text)


def normalize_string(s: str) -> str:
    """Та же нормализация, что у DocFacts.folded, для строки из ответа модели."""
    s = fold_homoglyphs(fix_ocr_digits(s)).casefold()
    return " ".join(s.split())


def _quant(d: Decimal) -> Decimal:
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _parse_numbers(text: str) -> set[Decimal]:
    out: set[Decimal] = set()
    for m in _NUMBER.finditer(text):
        tok = m.group(0)
        for ch in _SPACES:
            tok = tok.replace(ch, "")
        variants = [tok.replace(",", ".")]
        # «1,500» / «1.500» может быть и разделителем тысяч
        frac = re.search(r"[.,](\d+)$", tok)
        if frac and len(frac.group(1)) == 3:
            variants.append(re.sub(r"[.,]", "", tok))
        for v in variants:
            try:
                out.add(_quant(Decimal(v)))
            except InvalidOperation:
                pass
    return out


def _parse_dates(text: str) -> set[date]:
    out: set[date] = set()

    def add(y: int, m: int, d: int) -> None:
        try:
            out.add(date(y, m, d))
        except ValueError:
            pass

    for d, m, y in _DATE_NUMERIC.findall(text):
        add(int(y), int(m), int(d))
    for y, m, d in _DATE_ISO.findall(text):
        add(int(y), int(m), int(d))
    for d, stem, y in _DATE_WORDS.findall(text):
        add(int(y), _MONTHS[stem], int(d))
    return out


@dataclass
class DocFacts:
    text: str
    folded: str
    numbers: set[Decimal]
    dates: set[date]
    raw_numbers: set[Decimal]  # без исправления OCR: отличает ocr_corrected от ok

    def has_number(self, x: float) -> bool:
        return _quant(Decimal(str(x))) in self.numbers

    def has_raw_number(self, x: float) -> bool:
        return _quant(Decimal(str(x))) in self.raw_numbers


def analyze(text: str) -> DocFacts:
    fixed = fix_ocr_digits(text)
    return DocFacts(
        text=text,
        folded=normalize_string(text),
        numbers=_parse_numbers(fixed),
        dates=_parse_dates(normalize_string(text)),
        raw_numbers=_parse_numbers(text),
    )
