import json
from pathlib import Path

from doc_extractor.llm import ScriptedClient
from doc_extractor.pipeline import extract

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data/docs"


def text_of(n: str) -> str:
    return (DOCS / f"doc_{n}.txt").read_text(encoding="utf-8")


def gold(n: str) -> dict:
    d = json.loads((ROOT / f"data/gold/doc_{n}.json").read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_") and k != "expect_warnings"}


def run(text: str, answer: dict):
    return extract(text, ScriptedClient([json.dumps(answer, ensure_ascii=False)])).result


def test_grounded_fields_have_full_confidence():
    r = run(text_of("04"), gold("04"))
    conf = json.loads(r.model_dump_json())["field_confidence"]
    assert set(conf) == {
        "doc_type", "supplier_name", "supplier_bin", "doc_number", "doc_date",
        "currency", "total_amount", "vat_amount", "line_items",
    }
    assert all(v == 1.0 and isinstance(v, float) for v in conf.values())


def test_ocr_corrected_bin_has_lower_confidence():
    r = run(text_of("03"), gold("03") | {"supplier_bin": "210640900180"})
    assert r.supplier_bin == "210640900180"
    assert r.field_confidence["supplier_bin"] == 0.7


def test_bad_checksum_bin_has_half_confidence():
    r = run(text_of("01"), gold("01"))
    assert r.supplier_bin == "170340021554"
    assert r.field_confidence["supplier_bin"] == 0.5


def test_null_fields_absent_from_confidence():
    r = run(text_of("02"), gold("02"))
    for f in ("supplier_bin", "doc_number", "total_amount", "vat_amount"):
        assert getattr(r, f) is None
        assert f not in r.field_confidence
    assert r.field_confidence["doc_type"] == 1.0
    assert r.field_confidence["line_items"] == 1.0


def test_rejected_value_absent_from_confidence():
    r = run(text_of("09"), gold("09") | {"total_amount": 999999})
    assert r.total_amount is None
    assert "total_amount" not in r.field_confidence
    assert r.field_confidence == {}


def test_empty_result_has_empty_confidence():
    client = ScriptedClient(["{", '{"doc_type": "счёт"}'])
    ex = extract(text_of("01"), client)
    assert ex.result.field_confidence == {}
    assert ex.validation_failed is True


def test_no_line_items_means_no_line_items_key():
    r = run(text_of("04"), gold("04") | {"line_items": []})
    assert "line_items" not in r.field_confidence
    assert r.field_confidence["total_amount"] == 1.0
