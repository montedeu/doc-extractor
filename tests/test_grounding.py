import json
from pathlib import Path

from doc_extractor.llm import ScriptedClient
from doc_extractor.normalize import analyze, fix_ocr_digits, fold_homoglyphs
from doc_extractor.pipeline import extract

ROOT = Path(__file__).resolve().parent.parent


def text_of(n: str) -> str:
    return (ROOT / f"data/docs/doc_{n}.txt").read_text(encoding="utf-8")


def gold(n: str) -> dict:
    d = json.loads((ROOT / f"data/gold/doc_{n}.json").read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_") and k != "expect_warnings"}


def run(text: str, answer: dict):
    return extract(text, ScriptedClient([json.dumps(answer, ensure_ascii=False)])).result


def test_invented_total_becomes_null():
    a = gold("02") | {"total_amount": 510000}
    r = run(text_of("02"), a)
    assert r.total_amount is None
    assert "not_in_text: total_amount=510000" in r.warnings


def test_computed_vat_becomes_null():
    text = "Счёт № 5 от 12.11.2026\nИтого: 100 000 тенге\n"
    a = gold("01") | {"vat_amount": 16000, "total_amount": 100000}
    r = run(text, a)
    assert r.vat_amount is None
    assert "not_in_text: vat_amount=16000" in r.warnings


def test_vat_exempt_gives_zero():
    r = run(text_of("10"), gold("10"))
    assert r.vat_amount == 0
    assert not any("vat_amount" in w for w in r.warnings)


def test_vat_zero_without_phrase_becomes_null():
    r = run(text_of("02"), gold("02") | {"vat_amount": 0})
    assert r.vat_amount is None
    assert any("vat_amount" in w for w in r.warnings)


def test_currency_without_marker_becomes_null():
    text = "Счёт № 5 от 12.11.2026\nИтого: 100 000\n"
    r = run(text, gold("01") | {"currency": "KZT", "total_amount": 100000})
    assert r.currency is None
    assert "not_in_text: currency=KZT" in r.warnings


def test_today_date_not_in_text_becomes_null():
    r = run(text_of("09"), gold("09") | {"doc_date": "2026-11-19"})
    assert r.doc_date is None
    assert "not_in_text: doc_date=2026-11-19" in r.warnings


def test_kazakh_month_date_is_grounded():
    r = run(text_of("03"), gold("03"))
    assert str(r.doc_date) == "2026-10-09"
    assert not any("doc_date" in w for w in r.warnings)


def test_ocr_number_is_grounded():
    r = run(text_of("07"), gold("07"))
    assert r.total_amount == 39440
    assert r.line_items[1].qty == 2
    assert r.line_items[1].price == 4500
    assert not any(w.startswith("not_in_text") for w in r.warnings)


def test_comma_decimal_is_grounded():
    r = run(text_of("05"), gold("05"))
    assert r.vat_amount == 48275.86
    assert r.line_items[0].price == 280000.0


def test_invented_doc_number_becomes_null():
    r = run(text_of("01"), gold("01") | {"doc_number": "9999"})
    assert r.doc_number is None
    assert "not_in_text: doc_number=9999" in r.warnings


def test_clean_document_has_no_warnings():
    r = run(text_of("01"), gold("01"))
    assert not any(w.startswith("not_in_text") for w in r.warnings)
    assert r.total_amount == 186180


def test_missing_qty_nulls_subfield_keeps_item():
    a = gold("01")
    a["line_items"][0]["qty"] = 77
    r = run(text_of("01"), a)
    assert r.line_items[0].qty is None
    assert r.line_items[0].price == 2100
    assert "not_in_text: line_items[0].qty=77" in r.warnings


def test_latin_doc_number_matches_cyrillic_answer():
    r = run(text_of("03"), gold("03") | {"doc_number": "АКТ-09"})
    assert r.doc_number == "АКТ-09"


def test_number_formats_and_spaces():
    f = analyze("1 000 000 и 25 000 и 7 500 и 713.40 и 280 000,00")
    for x in (1000000, 25000, 7500, 713.4, 280000):
        assert f.has_number(x)
    assert not f.has_number(1)
    assert not f.has_number(0)


def test_date_formats():
    from datetime import date
    f = analyze("12/11/2026; 15 қараша 2026; 12 ноября 2026 г.; 2l.lO.2O26")
    assert {date(2026, 11, 12), date(2026, 11, 15), date(2026, 10, 21)} <= f.dates


def test_normalize_helpers():
    assert fold_homoglyphs("AKT") == "АКТ"
    assert fix_ocr_digits("25 OOO") == "25 000"
    assert fix_ocr_digits("Оплата Olga") == "Оплата Olga"
