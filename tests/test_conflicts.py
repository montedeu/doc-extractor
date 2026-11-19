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


def test_contract_date_is_not_creation_date():
    assert warnings_of("Счёт № 5 от 01.02.2026\nДоговор № 3 от 10.01.2025", "conflicting_dates") == []
    assert warnings_of("Счёт от 01.02.2026 по договору от 10.01.2025", "conflicting_dates") == []


def test_payment_terms_are_not_totals():
    text = "К оплате: 5 дней\nИтого к оплате: 100 000"
    assert warnings_of(text, "conflicting_totals") == []


def test_vat_line_does_not_hide_words_mismatch():
    text = (
        "Итого НДС: 12 000\nВсего к оплате: 112 000\n"
        "Сумма прописью: двенадцать тысяч тенге"
    )
    assert len(warnings_of(text, "amount_words_mismatch")) == 1


def test_fractional_digits_compare_integer_part():
    text = "Итого: 1 000,50\nСумма прописью: пять тысяч тенге"
    assert len(warnings_of(text, "amount_words_mismatch")) == 1


def test_subtotal_does_not_trigger_words_mismatch():
    text = (
        "Итого: 128 400\nНДС 12%: 15 408\n"
        "Всего к оплате: Сто сорок три тысячи восемьсот восемь тенге 00 тиын"
    )
    assert warnings_of(text, "amount_words_mismatch") == []


def test_bare_scale_word_is_not_amount():
    text = "Сумма, тысяч тенге\nИтого: 100"
    assert warnings_of(text, "amount_words_mismatch") == []


def test_period_and_payment_dates_not_conflicts():
    a = "Акт № 5 от 31.10.2026 за период от 01.10.2026 по 31.10.2026"
    b = "Счёт на оплату № 5 от 31.10.2026 (оплата от 01.11.2026)"
    assert warnings_of(a, "conflicting_dates") == []
    assert warnings_of(b, "conflicting_dates") == []


def test_prepayment_not_conflict():
    text = "К оплате (предоплата 50%): 74 472 тенге\nВсего к оплате: 148 944"
    assert warnings_of(text, "conflicting_totals") == []


def test_number_from_needs_document_title_before_it():
    text = (
        "Счёт № 5 от 01.02.2026\nПлатёжное поручение № 3 от 10.02.2026\n"
        "Основание: протокол № 4 от 01.01.2025"
    )
    assert warnings_of(text, "conflicting_dates") == []


def test_prepayment_excluded_from_fallback_words_check():
    text = "Итого: 1000\nК оплате аванс: 500\nпятьсот тенге"
    assert len(warnings_of(text, "amount_words_mismatch")) == 1
