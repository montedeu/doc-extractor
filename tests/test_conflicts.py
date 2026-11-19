import json
from pathlib import Path

from doc_extractor.llm import ScriptedClient
from doc_extractor.pipeline import extract

ROOT = Path(__file__).resolve().parent.parent
EMPTY = json.dumps(
    {
        "doc_type": "invoice", "supplier_name": None, "supplier_bin": None,
        "doc_number": None, "doc_date": None, "currency": None,
        "total_amount": None, "vat_amount": None, "line_items": [],
    }
)


def warnings_of(text: str, prefix: str) -> list[str]:
    result = extract(text, ScriptedClient([EMPTY])).result
    return [w for w in result.warnings if w.startswith(prefix)]


def doc(n: str) -> str:
    return (ROOT / f"data/docs/doc_{n}.txt").read_text(encoding="utf-8")


def test_two_payable_totals_give_warning():
    assert warnings_of(doc("06"), "conflicting_totals") == [
        "conflicting_totals: 148944 / 149844"
    ]


def test_subtotal_and_total_are_not_conflict():
    assert warnings_of(doc("01"), "conflicting_totals") == []
    assert warnings_of(doc("04"), "conflicting_totals") == []


def test_same_total_twice_is_not_conflict():
    text = "Всего к оплате: 100 000 тенге\nК оплате: 100 000 тенге"
    assert warnings_of(text, "conflicting_totals") == []


def test_words_match_digits_no_warning():
    assert warnings_of(doc("05"), "amount_words_mismatch") == []


def test_words_differ_from_digits_give_warning():
    text = "Итого к оплате: 340 000 тенге\nСумма прописью: Триста пятьдесят тысяч тенге"
    assert warnings_of(text, "amount_words_mismatch") == [
        "amount_words_mismatch: прописью 350000, цифрами 340000"
    ]


def test_fractional_digits_skip_words_check():
    text = "Итого к оплате: 340 000,50 тенге\nСумма прописью: Триста пятьдесят тысяч тенге"
    assert warnings_of(text, "amount_words_mismatch") == []


def test_words_only_total_no_warning():
    for prefix in ("conflicting_totals", "amount_words_mismatch", "conflicting_dates"):
        assert warnings_of(doc("08"), prefix) == []


def test_validity_and_due_dates_are_not_conflicts():
    assert warnings_of(doc("02"), "conflicting_dates") == []
    assert warnings_of(doc("10"), "conflicting_dates") == []


def test_two_creation_dates_give_warning():
    text = "Счёт № 5 от 01.11.2026\nПоставщик ТОО\nДата составления: 03.11.2026"
    assert warnings_of(text, "conflicting_dates") == [
        "conflicting_dates: 01.11.2026 / 03.11.2026"
    ]


def test_two_creation_dates_in_one_line():
    text = "Счёт № 5 от 01.11.2026 ... Дата составления: 03.11.2026"
    assert warnings_of(text, "conflicting_dates") != []
