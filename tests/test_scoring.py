from collections import Counter

from doc_extractor.scoring import compare_field, evaluate, format_report


def item(name, qty=1, price=10):
    return {"name": name, "qty": qty, "price": price}


def doc(**kw):
    base = {
        "doc_type": "invoice", "supplier_name": None, "supplier_bin": None,
        "doc_number": None, "doc_date": None, "currency": None,
        "total_amount": None, "vat_amount": None, "line_items": [], "warnings": [],
    }
    return {**base, **kw}


def gold(**kw):
    return {**doc(), "expect_warnings": False, **kw}


def run(res, gld):
    return evaluate({"d": res}, {"d": gld})


def test_four_outcomes():
    assert compare_field("doc_number", None, None) == "correct"
    assert compare_field("doc_number", "1", None) == "hallucination"
    assert compare_field("doc_number", None, "1") == "miss"
    assert compare_field("doc_number", "1", "2") == "wrong"


def test_null_vs_null_counts_as_correct_in_accuracy():
    r = run(doc(), gold())
    assert r.outcomes["doc_number"]["correct"] == 1
    assert r.accuracy("doc_number") == 1.0


def test_amounts_compared_exactly_after_rounding():
    assert compare_field("total_amount", 713.4, 713.40) == "correct"
    assert compare_field("total_amount", 48275.85, 48275.86) == "wrong"


def test_supplier_names_normalized():
    f = "supplier_name"
    assert compare_field(f, 'TOO "Steppe Cloud Services"', 'ТОО «Steppe Cloud Services»') == "correct"
    assert compare_field(f, 'T0O "Бepeкe Пpинт"', 'ТОО "Береке Принт"') == "correct"


def test_doc_type_has_no_hallucination():
    assert compare_field("doc_type", "invoice", "unknown") == "wrong"
    assert compare_field("doc_type", "unknown", "invoice") == "wrong"


def test_line_items_matched_by_name_not_position():
    A, B, C = item("Альфа"), item("Бета"), item("Гамма")
    r = run(doc(line_items=[B, C]), gold(line_items=[A, B, C]))
    assert r.outcomes["line_items.name"] == Counter(correct=2, miss=1)
    assert r.outcomes["line_items.qty"] == Counter(correct=2, miss=1)
    assert r.outcomes["line_items.price"] == Counter(correct=2, miss=1)


def test_extra_line_item_is_hallucination():
    A = item("Альфа")
    r = run(doc(line_items=[A, item("Выдумка")]), gold(line_items=[A]))
    assert r.outcomes["line_items.name"] == Counter(correct=1, hallucination=1)


def test_order_metric():
    A, B = item("Альфа"), item("Бета")
    r = run(doc(line_items=[B, A]), gold(line_items=[A, B]))
    assert (r.order_ok, r.order_total) == (0, 1)
    assert r.outcomes["line_items.name"] == Counter(correct=2)
    r = run(doc(line_items=[A, B]), gold(line_items=[A, B]))
    assert (r.order_ok, r.order_total) == (1, 1)


def test_warnings_expectation():
    r = run(doc(warnings=[]), gold(expect_warnings=True))
    assert (r.warnings_ok, r.warnings_total) == (0, 1)
    r = run(doc(warnings=[]), gold(expect_warnings=False))
    assert (r.warnings_ok, r.warnings_total) == (1, 1)


def test_report_lists_hallucinations_and_misses_separately():
    r = evaluate(
        {"a": doc(doc_number="1"), "b": doc()},
        {"a": gold(), "b": gold(doc_number="2")},
    )
    text = format_report(r)
    assert "галлюцинации" in text and "пропуски" in text and "неверно" in text
    row = next(l for l in text.splitlines() if l.startswith("doc_number"))
    assert row.split()[1:5] == ["0", "1", "1", "0"]
    total = next(l for l in text.splitlines() if l.startswith("ИТОГО"))
    assert total.split()[1:5] == ["14", "1", "1", "0"]  # 16 скалярных исходов, 2 из них не «верно»
