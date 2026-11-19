"""Поиск противоречий внутри документа (ADR 0003): код только сообщает."""

import re
from datetime import date
from decimal import Decimal

from doc_extractor.normalize import (
    NUMBER_RE,
    SPACES_RE,
    DocFacts,
    normalize_string,
    number_variants,
    parse_dates,
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
_NOT_AMOUNT_TAIL = re.compile(r"\s*(?:%|тиын|тийін|дн|день|дней|календар|банковск|рабоч|шт|мес)")  # процент, тиыны прописью
_NOT_WORDS_LINE = re.compile(r"ндс|ққс|наименован|позици")
_PREPAYMENT = re.compile(r"предоплат|аванс|частичн")
# «№ <номер> от <дата>»: «от» сразу после номера документа
_AFTER_NUMBER = re.compile(
    r"(?:сч[её]т|шот|акт|накладн|предложени|\bкп\b)[^№]*№\s*[^\s№]+\s*$"
)
# «Счёт от <дата>»: «от» после названия документа в начале строки (только слова, не больше 4)
_AFTER_TITLE = re.compile(
    r"^\s*(?:сч[её]т|шот|акт|накладн|предложени|коммерческ)[а-яёәіңғүұқөһ-]*(?:\s+[а-яёәіңғүұқөһ-]+){0,3}\s*$"
)
_REF_ANYWHERE = re.compile(r"договор|контракт|доверенност|спецификац|заказ|приложени")
_NOT_CREATION = re.compile(r"срок|действительн|действует|действ\.")


def _amount_after(line: str, start: int) -> Decimal | None:
    """Первое число после метки, не процент."""
    for m in NUMBER_RE.finditer(line, start):
        if _NOT_AMOUNT_TAIL.match(line[m.end():]):
            continue
        tok = re.sub(f"[{SPACES_RE}]", "", m.group(0))
        # неоднозначное «148.944»: для сумм читаем как группу тысяч
        return Decimal(min(number_variants(tok), key=len))
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


def _payable_lines(lines: list[str]) -> list[str]:
    """Строки «к оплате» без предоплаты/аванса: частичная сумма не итог."""
    return [x for x in lines if _PAYABLE.search(x) and not _PREPAYMENT.search(x)]


def _from_creation(prefix: str) -> bool:
    """«от» — дата составления, только если стоит сразу после номера или названия документа."""
    if _REF_ANYWHERE.search(prefix):
        return False
    if _AFTER_NUMBER.search(prefix):
        # «накладная № 7 от …» не в начале строки — ссылка на другой документ
        ref = re.search(r"накладн", prefix)
        return not ref or ref.start() == len(prefix) - len(prefix.lstrip())
    return bool(_AFTER_TITLE.match(prefix)) and not re.search(r"накладн", prefix.lstrip()[1:])


def _conflicting_totals(lines: list[str]) -> list[str]:
    amounts = sorted(set(_total_amounts(_payable_lines(lines), _PAYABLE)))
    if len(amounts) < 2:
        return []
    return ["conflicting_totals: " + " / ".join(_fmt(a) for a in amounts)]


def _words_mismatch(facts: DocFacts, lines: list[str]) -> list[str]:
    if not facts.amounts_in_words:
        return []
    lines = [x for x in lines if not _NOT_WORDS_LINE.search(x) and not _PREPAYMENT.search(x)]
    # промежуточное «Итого» не участвует, если в документе есть строка «к оплате»
    payable = _payable_lines(lines)
    digits = _total_amounts(payable, _PAYABLE) if payable else _total_amounts(lines, _TOTAL)
    if not digits:
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
        prefix = text[line_start : m.start()]
        if _NOT_CREATION.search(prefix):
            continue
        if m.group(0).startswith("от") and not _from_creation(prefix):
            continue
        found |= parse_dates(m.group(1) or m.group(2))
    if len(found) < 2:
        return []
    return ["conflicting_dates: " + " / ".join(d.strftime("%d.%m.%Y") for d in sorted(found))]


def find_conflicts(facts: DocFacts) -> list[str]:
    lines = facts.lines
    text = "\n".join(lines)
    return _conflicting_totals(lines) + _words_mismatch(facts, lines) + _conflicting_dates(text)
