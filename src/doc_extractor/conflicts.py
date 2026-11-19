"""Поиск противоречий внутри документа (ADR 0003): код только сообщает."""

import re
from datetime import date
from decimal import Decimal

from doc_extractor.normalize import (
    _NUMBER,
    _SPACES,
    DocFacts,
    _parse_dates,
    _variants,
    normalize_string,
)

_PAYABLE = re.compile(r"к оплате|төлеуге")
_TOTAL = re.compile(r"итого|всего|к оплате|барлығы|төлеуге")
_DATE = (
    r"(?:\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{4}-\d{2}-\d{2}"
    r"|[«\"(\[]?\d{1,2}[»\")\]]?\s+[а-яёәіңғүұқөһ]+\s+\d{4})"
)
_CREATION = re.compile(
    rf"(?<![а-яёәіңғүұқөһ])(?:от|дата составления|составлено|составлен)\s*:?\s*({_DATE})"
    rf"|^\s*дата\s*:?\s*({_DATE})",
    re.MULTILINE,
)
_NOT_AMOUNT_TAIL = re.compile(r"\s*(?:%|тиын|тийін)")  # процент, тиыны прописью
_NOT_CREATION = re.compile(r"срок|действительн|действует|действ\.")


def _amount_after(line: str, start: int) -> Decimal | None:
    """Первое число после метки, не процент."""
    for m in _NUMBER.finditer(line, start):
        if _NOT_AMOUNT_TAIL.match(line[m.end():]):
            continue
        tok = re.sub(f"[{_SPACES}]", "", m.group(0))
        # неоднозначное «148.944»: для сумм читаем как группу тысяч
        return Decimal(min(_variants(tok), key=len))
    return None


def _total_amounts(folded_lines: list[str], label: re.Pattern) -> list[Decimal]:
    out: list[Decimal] = []
    for line in folded_lines:
        m = label.search(line)
        if not m:
            continue
        amount = _amount_after(line, m.start())
        if amount is not None:
            out.append(amount)
    return out


def _fmt(d: Decimal) -> str:
    return str(int(d)) if d == d.to_integral_value() else str(d)


def _conflicting_totals(lines: list[str]) -> list[str]:
    amounts = sorted(set(_total_amounts(lines, _PAYABLE)))
    if len(amounts) < 2:
        return []
    return ["conflicting_totals: " + " / ".join(_fmt(a) for a in amounts)]


def _words_mismatch(facts: DocFacts, lines: list[str]) -> list[str]:
    if not facts.amounts_in_words:
        return []
    digits = _total_amounts(lines, _TOTAL)
    if not digits or any(d != d.to_integral_value() for d in digits):
        return []
    have = {int(d) for d in digits}
    bad = [w for w in dict.fromkeys(facts.amounts_in_words) if w not in have]
    if not bad:
        return []
    return [
        "amount_words_mismatch: прописью "
        + " / ".join(str(w) for w in bad)
        + ", цифрами "
        + " / ".join(str(n) for n in sorted(have))
    ]


def _conflicting_dates(text: str) -> list[str]:
    found: set[date] = set()
    for m in _CREATION.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        if _NOT_CREATION.search(text[line_start : m.start()]):
            continue
        found |= _parse_dates(m.group(1) or m.group(2))
    if len(found) < 2:
        return []
    return ["conflicting_dates: " + " / ".join(d.strftime("%d.%m.%Y") for d in sorted(found))]


def find_conflicts(facts: DocFacts) -> list[str]:
    lines = [normalize_string(line) for line in facts.text.splitlines()]
    text = "\n".join(lines)
    return _conflicting_totals(lines) + _words_mismatch(facts, lines) + _conflicting_dates(text)
