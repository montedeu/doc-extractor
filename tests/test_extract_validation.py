import json
from pathlib import Path

import pytest

from doc_extractor.llm import LLMUnavailableError, ScriptedClient
from doc_extractor.pipeline import extract
from doc_extractor.prompt import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / "data/docs/doc_01.txt").read_text(encoding="utf-8")

VALID = {
    "doc_type": "invoice",
    "supplier_name": 'ТОО "Алматы Трейд Компани"',
    "supplier_bin": "170340021554",
    "doc_number": "1245",
    "doc_date": "2026-11-12",
    "currency": "KZT",
    "total_amount": 186180,
    "vat_amount": 25680,
    "line_items": [{"name": "Бумага А4 80г/м2", "qty": 50, "price": 2100}],
}
VALID_JSON = json.dumps(VALID, ensure_ascii=False)


def assert_empty(result):
    assert result.doc_type == "unknown"
    assert result.line_items == []
    for f in ("supplier_name", "supplier_bin", "doc_number", "doc_date",
              "currency", "total_amount", "vat_amount"):
        assert getattr(result, f) is None


def test_valid_answer_becomes_result():
    client = ScriptedClient([VALID_JSON])
    ex = extract(TEXT, client)
    dumped = json.loads(ex.result.model_dump_json())
    for key, value in VALID.items():
        assert dumped[key] == value
    assert dumped["doc_date"] == "2026-11-12"
    assert dumped["field_confidence"] == {
        "doc_type": 1.0, "supplier_name": 1.0, "supplier_bin": 0.5, "doc_number": 1.0,
        "doc_date": 1.0, "currency": 1.0, "total_amount": 1.0, "vat_amount": 1.0,
        "line_items": 1.0,
    }
    assert dumped["warnings"] == ["bin_checksum_failed: supplier_bin=170340021554"]
    assert ex.validation_failed is False
    assert len(client.calls) == 1


def test_non_json_answer_is_retried_with_error_text():
    client = ScriptedClient(["это не json", VALID_JSON])
    ex = extract(TEXT, client)
    assert ex.result.doc_number == "1245"
    assert ex.validation_failed is False
    assert len(client.calls) == 2
    retry = " ".join(m["content"] for m in client.calls[1])
    assert "это не json" in retry
    assert "json_invalid" in retry or "Invalid JSON" in retry
    assert len(client.calls[1]) > len(client.calls[0])


def test_schema_violation_is_retried():
    bad = json.dumps({**VALID, "supplier_bin": "17034002155"})
    client = ScriptedClient([bad, VALID_JSON])
    ex = extract(TEXT, client)
    assert ex.result.supplier_bin == "170340021554"
    assert len(client.calls) == 2
    assert "supplier_bin" in " ".join(m["content"] for m in client.calls[1])


def test_two_invalid_answers_give_empty_result():
    client = ScriptedClient(["{", '{"doc_type": "счёт"}'])
    ex = extract(TEXT, client)
    assert_empty(ex.result)
    assert len(ex.result.warnings) == 1
    assert ex.result.warnings[0].startswith("validation_failed:")
    assert ex.validation_failed is True
    assert len(client.calls) == 2


def test_extra_field_is_rejected():
    bad = json.dumps({**VALID, "comment": "hi"})
    client = ScriptedClient([bad, bad])
    ex = extract(TEXT, client)
    assert ex.validation_failed is True
    assert_empty(ex.result)


def test_empty_document_skips_model():
    client = ScriptedClient([])
    ex = extract("  \n\t", client)
    assert_empty(ex.result)
    assert ex.result.warnings == ["empty_document"]
    assert ex.validation_failed is False
    assert client.calls == []


def test_prompt_has_no_test_set_examples():
    assert "Алматы Трейд" in TEXT
    assert "Алматы Трейд" not in SYSTEM_PROMPT
    assert "170340021554" not in SYSTEM_PROMPT
    for path in (ROOT / "data/docs").glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if len(line) >= 15:
                assert line not in SYSTEM_PROMPT


def test_prompt_has_spec_rules():
    assert "Если название поставщика двуязычное, бери русский вариант" in SYSTEM_PROMPT
    assert "без НДС" in SYSTEM_PROMPT
    assert "vat_amount = 0" in SYSTEM_PROMPT
    assert "l, I вместо 1" in SYSTEM_PROMPT
    assert "O, О вместо 0" in SYSTEM_PROMPT
    assert "организационно-правовой" in SYSTEM_PROMPT and "ЖШС → ТОО" in SYSTEM_PROMPT
    assert "целиком" in SYSTEM_PROMPT


def test_markdown_fence_is_stripped():
    client = ScriptedClient(["```json\n" + VALID_JSON + "\n```"])
    ex = extract(TEXT, client)
    assert ex.validation_failed is False
    assert len(client.calls) == 1


@pytest.mark.parametrize("patch", [
    {"currency": "тг"},
    {"doc_type": "Счёт на оплату"},
    {"total_amount": -1},
    {"total_amount": "Infinity"},
    {"vat_amount": "-Infinity"},
])
def test_invalid_values_are_retried(patch):
    client = ScriptedClient([json.dumps({**VALID, **patch}), VALID_JSON])
    ex = extract(TEXT, client)
    assert ex.validation_failed is False
    assert len(client.calls) == 2


def test_nan_is_retried():
    bad = VALID_JSON.replace('"total_amount": 186180', '"total_amount": NaN')
    assert "NaN" in bad
    client = ScriptedClient([bad, VALID_JSON])
    ex = extract(TEXT, client)
    assert ex.validation_failed is False
    assert len(client.calls) == 2


def test_llm_unavailable_propagates():
    class Down:
        def complete(self, messages):
            raise LLMUnavailableError("down")

    with pytest.raises(LLMUnavailableError):
        extract(TEXT, Down())
