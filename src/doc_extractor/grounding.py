"""Сверка с текстом (ADR 0001): значение, которого нет в документе, становится null."""

import re
from dataclasses import dataclass
from typing import Callable, Literal

from doc_extractor.normalize import DocFacts, normalize_string
from doc_extractor.schema import ModelAnswer

FieldStatus = Literal["ok", "ocr_corrected", "warned"]

_VAT_WORD = re.compile(r"ндс|ққс")
_VAT_EXEMPT = re.compile(
    r"без\s+ндс|ндс\s+не\s+облагается|не\s+облагается\s+ндс"
    r"|ққс\s*-?\s*сыз|ққс\s+салынбайды"
)
_L = r"[а-яёәіңғүұқөһa-z0-9]"
_CURRENCY = {
    "KZT": re.compile(rf"(?<!{_L})(тенге|тг|теңге)(?!{_L})|₸|(?<!{_L})kzt(?!{_L})"),
    "USD": re.compile(rf"(?<!{_L})usd(?!{_L})|\$|долл"),
    "EUR": re.compile(rf"(?<!{_L})eur(?!{_L})|€|евро"),
}


@dataclass
class Grounded:
    answer: ModelAnswer
    warnings: list[str]
    status: dict[str, FieldStatus]


def _fmt(v: object) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)


def _currency_in_text(code: str, facts: DocFacts) -> bool:
    pat = _CURRENCY[code]
    return bool(pat.search(facts.text.casefold()) or pat.search(facts.folded))


def _number_status(x: float | None, facts: DocFacts) -> FieldStatus | None:
    if x is None or not facts.has_number(x):
        return None
    return "ok" if facts.has_raw_number(x) else "ocr_corrected"


def _doc_number_status(value: str, facts: DocFacts) -> FieldStatus | None:
    norm = normalize_string(value)
    if not norm or not re.search(rf"(?<!{_L}){re.escape(norm)}(?!{_L})", facts.folded):
        return None
    raw = " ".join(value.casefold().split())
    return "ok" if raw in " ".join(facts.text.casefold().split()) else "ocr_corrected"


def ground(answer: ModelAnswer, facts: DocFacts) -> Grounded:
    a = answer.model_copy(deep=True)
    warnings: list[str] = []
    status: dict[str, FieldStatus] = {}

    def check(
        name: str, value: object, st: FieldStatus | None, nullify: Callable[[], None]
    ) -> None:
        if value is None:
            return
        if st is None:
            warnings.append(f"not_in_text: {name}={_fmt(value)}")
            status[name] = "warned"
            nullify()
        else:
            status[name] = st

    check("total_amount", a.total_amount, _number_status(a.total_amount, facts),
          lambda: setattr(a, "total_amount", None))

    if a.vat_amount is not None:
        if a.vat_amount == 0:
            vat_st: FieldStatus | None = "ok" if _VAT_EXEMPT.search(facts.folded) else None
        elif _VAT_WORD.search(facts.folded):
            vat_st = _number_status(a.vat_amount, facts)
        else:
            vat_st = None
        check("vat_amount", a.vat_amount, vat_st, lambda: setattr(a, "vat_amount", None))

    check("doc_date", a.doc_date, "ok" if a.doc_date in facts.dates else None,
          lambda: setattr(a, "doc_date", None))

    if a.currency is not None:
        check("currency", a.currency,
              "ok" if _currency_in_text(a.currency, facts) else None,
              lambda: setattr(a, "currency", None))

    if a.doc_number is not None:
        check("doc_number", a.doc_number, _doc_number_status(a.doc_number, facts),
              lambda: setattr(a, "doc_number", None))

    for i, item in enumerate(a.line_items):
        for f in ("qty", "price"):
            v = getattr(item, f)
            check(f"line_items[{i}].{f}", v, _number_status(v, facts),
                  lambda item=item, f=f: setattr(item, f, None))

    return Grounded(a, warnings, status)
