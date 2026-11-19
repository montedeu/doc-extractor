"""Сверка с текстом (ADR 0001): значение, которого нет в документе, становится null."""

import re
from dataclasses import dataclass
from typing import Callable, Literal

from doc_extractor.bin_check import bin_checksum_ok
from doc_extractor.normalize import (
    BinCandidate,
    DocFacts,
    fold_homoglyphs,
    name_core,
    name_core_text,
    normalize_string,
)
from doc_extractor.schema import ModelAnswer

FieldStatus = Literal["ok", "ocr_corrected", "warned"]

_VAT_WORD = re.compile(r"ндс|ққс")
_VAT_EXEMPT = re.compile(
    r"ндс\s+не\s+облагается|не\s+облагается\s+ндс|освобожд\w*\s+от\s+ндс"
    r"|ққс\s*-?\s*сыз|ққс\s+салынбайды"
    r"|(?<![^\W\d_])без\s+ндс(?!.*\d)"  # «Без НДС» без числа дальше в той же строке
)
_L = r"[а-яёәіңғүұқөһa-z0-9]"
_B = r"(?<![^\W\d_])(?<!-)"  # слева не буква и не дефис
_E = r"(?![^\W\d_])(?!-)"
_CURRENCY = {
    "KZT": re.compile(rf"{_B}(?:тенге|тг|теңге|kzt){_E}|₸"),
    "USD": re.compile(rf"{_B}(?:usd|долл(?:ар(?:ов|а|ы)?)?){_E}|\$"),
    "EUR": re.compile(rf"{_B}(?:eur|евро){_E}|€"),
}


_TOTALS_LABEL = re.compile(
    r"(?:(?:итого|всего)(?: по документу| к оплате| (?:без|с) ндс)?|к оплате|в том числе ндс"
    r"|ндс|ққс|барлығы(?: төлеуге)?|жиыны|жалпы сомасы)"
    r"(?: [\d%]+)*(?: тенге| тг)?"
)
_PARTS_SPLIT = re.compile(r"\s+/\s+")


_NL = r"(?<![^\W\d_])"  # слева не буква
_NR = r"(?![^\W\d_])"  # справа не буква
# Тексты сравниваются после normalize_string: латинские B/b -> кириллическая «в», I остаётся.
_BIN_ALT = r"(?:б[иіil1]н|в[іil1]n|бсн|вsn)"
_IIN_ALT = r"(?:иин|iin|іін|жсн)"
_BIN_LABEL = re.compile(rf"{_NL}{_BIN_ALT}{_NR}")
_IIN_LABEL = re.compile(rf"{_NL}{_IIN_ALT}{_NR}")
# «БИН/ИИН», «ИИН/БИН», «БИН (ИИН)»: тип по метке не решают, остаётся роль
_COMBINED = re.compile(
    rf"{_NL}(?:{_BIN_ALT}\s*[/(]\s*{_IIN_ALT}|{_IIN_ALT}\s*[/(]\s*{_BIN_ALT})\)?{_NR}"
)
_BUYER = re.compile(
    rf"{_NL}(?:покупател\w{{0,3}}|заказчик\w{{0,3}}|плательщик\w{{0,3}}"
    rf"|сатып\s+алушы\w{{0,3}}|тапсырыс\s+беруші\w{{0,3}}){_NR}"
)
_SUPPLIER = re.compile(
    rf"{_NL}(?:исполнител\w{{0,3}}|поставщик\w{{0,3}}|продав\w{{0,4}}"
    rf"|орындаушы\w{{0,3}}|жеткізуші\w{{0,3}}|сатушы\w{{0,3}}){_NR}"
)
_QUOTED = re.compile(r'«[^»]*»|"[^"]*"|“[^”]*”')


def _mask(line: str) -> str:
    """Названия в кавычках не содержат меток: гасим пробелами, длина сохраняется."""
    return _QUOTED.sub(lambda m: " " * len(m.group(0)), line)


def _last_pos(pat: re.Pattern, s: str) -> int:
    ms = list(pat.finditer(s))
    return ms[-1].start() if ms else -1


def _role(left: str) -> str | None:
    b, s = _last_pos(_BUYER, left), _last_pos(_SUPPLIER, left)
    if b < 0 and s < 0:
        return None
    return "buyer" if b > s else "supplier"


def _reject_reason(c: BinCandidate) -> str | None:
    """Ближайшая метка слева от числа решает; без меток слева — вся строка."""
    line = _mask(c.line)
    end = c.start + len(c.raw)
    left = _COMBINED.sub(" бин ", normalize_string(line[: c.start]))
    right = normalize_string(line[end:])
    whole = _COMBINED.sub(" бин ", normalize_string(line))
    b, i = _last_pos(_BIN_LABEL, left), _last_pos(_IIN_LABEL, left)
    if b >= 0 or i >= 0:
        is_iin = i > b
    else:
        is_iin = bool(_IIN_LABEL.search(whole)) and not _BIN_LABEL.search(whole)
    if is_iin:
        return "ИИН"
    role = _role(left)
    if role is None:
        # метка роли справа: «<БИН> — БИН покупателя»
        rb, rs = _BUYER.search(right), _SUPPLIER.search(right)
        if rb and (not rs or rb.start() < rs.start()):
            role = "buyer"
        elif rs:
            role = "supplier"
    if role is None:
        for prev in c.prev_lines:
            role = _role(normalize_string(_mask(prev)))
            if role:
                break
    return "покупатель" if role == "buyer" else None


def _rank(c: BinCandidate) -> int:
    ok = bin_checksum_ok(c.digits)
    if ok:
        return 1 if c.ocr_fixed else 0
    return 3 if c.ocr_fixed else 2


def _bin_status(a: ModelAnswer, facts: DocFacts, warnings: list[str]) -> FieldStatus | None:
    """None: значение обнулить (предупреждение уже добавлено)."""
    v = a.supplier_bin
    cands = [c for c in facts.bin_candidates if v in (c.digits, c.raw)]
    if not cands:
        warnings.append(f"not_in_text: supplier_bin={v}")
        return None
    reasons = [(c, _reject_reason(c)) for c in cands]
    if any(r == "покупатель" for _, r in reasons):
        warnings.append(f"bin_not_supplier: supplier_bin={v} (покупатель)")
        return None
    good = sorted((c for c, r in reasons if r is None), key=_rank)
    if not good:
        warnings.append(f"bin_not_supplier: supplier_bin={v} ({reasons[0][1]})")
        return None
    c = good[0]
    if not bin_checksum_ok(c.digits):
        warnings.append(f"bin_checksum_failed: supplier_bin={v}")
        # OCR-правка разрешена только при сходящейся контрольной цифре
        return None if c.ocr_fixed else "warned"
    if c.ocr_fixed:
        warnings.append(f"ocr_corrected: supplier_bin {c.raw} → {c.digits}")
        return "ocr_corrected"
    return "ok"


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
    # двойники (латиница/кириллица) не OCR-правка, как у названия
    raw = " ".join(fold_homoglyphs(value).casefold().split())
    text = " ".join(fold_homoglyphs(facts.text).casefold().split())
    return "ok" if raw in text else "ocr_corrected"


def _name_in_text(value: str, text_core: str) -> bool:
    core = name_core(value)
    if not core.strip("/ "):
        return False
    return bool(re.search(rf"(?<!\w){re.escape(core)}(?!\w)", text_core))


def _item_name_ok(name: str, text_core: str) -> bool:
    core = name_core(name)
    if not core.strip("/ ") or _TOTALS_LABEL.fullmatch(core):
        return False
    if _name_in_text(name, text_core):
        return True
    parts = [p for p in _PARTS_SPLIT.split(name) if name_core(p)]  # «Қазақша / Русский»
    return len(parts) > 1 and all(_name_in_text(p, text_core) for p in parts)


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
            vat_st: FieldStatus | None = (
                "ok" if any(_VAT_EXEMPT.search(x) for x in facts.lines) else None
            )
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

    text_core = name_core_text(facts.text)

    if a.supplier_name is not None:
        check("supplier_name", a.supplier_name,
              "ok" if _name_in_text(a.supplier_name, text_core) else None,
              lambda: setattr(a, "supplier_name", None))

    kept = []
    for item in a.line_items:
        if _item_name_ok(item.name, text_core):
            kept.append(item)
        else:
            warnings.append(f"line_item_dropped: {item.name}")
    a.line_items = kept

    if a.supplier_bin is not None:
        st = _bin_status(a, facts, warnings)
        if st is None:
            a.supplier_bin = None
        else:
            status["supplier_bin"] = st

    for i, item in enumerate(a.line_items):  # индексы по итоговому списку
        for f in ("qty", "price"):
            v = getattr(item, f)
            check(f"line_items[{i}].{f}", v, _number_status(v, facts),
                  lambda item=item, f=f: setattr(item, f, None))

    return Grounded(a, warnings, status)
