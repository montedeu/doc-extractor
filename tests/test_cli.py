import io
import json
from pathlib import Path

from doc_extractor.cli import main
from doc_extractor.llm import LLMUnavailableError, ScriptedClient

ROOT = Path(__file__).resolve().parent.parent
DOC_02 = ROOT / "data/docs/doc_02.txt"

VALID = {
    "doc_type": "commercial_offer",
    "supplier_name": "Digital Solutions",
    "supplier_bin": None,
    "doc_number": None,
    "doc_date": "2026-11-05",
    "currency": "KZT",
    "total_amount": None,
    "vat_amount": None,
    "line_items": [{"name": "Разработка сайта", "qty": None, "price": 450000}],
}
VALID_JSON = json.dumps(VALID, ensure_ascii=False)


class UnavailableClient:
    def complete(self, messages):
        raise LLMUnavailableError("connection refused")


def run(argv, client, stdin_text=""):
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, client=client, stdin=io.StringIO(stdin_text),
                stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def test_file_argument_prints_json_and_exits_0():
    code, out, err = run(["extract", str(DOC_02)], ScriptedClient([VALID_JSON]))
    assert code == 0
    assert json.loads(out)["doc_type"] == "commercial_offer"
    assert err == ""


def test_stdin_dash_reads_document():
    text = DOC_02.read_text(encoding="utf-8")
    client = ScriptedClient([VALID_JSON])
    code, out, _ = run(["extract", "-"], client, stdin_text=text)
    assert code == 0
    assert client.calls[0][-1]["content"] == text


def test_missing_file_exits_1_with_message(tmp_path):
    missing = tmp_path / "nope.txt"
    code, out, err = run(["extract", str(missing)], ScriptedClient([]))
    assert code == 1
    assert out == ""
    assert str(missing) in err


def test_non_utf8_file_exits_1(tmp_path):
    path = tmp_path / "cp1251.txt"
    path.write_bytes("Счёт на оплату".encode("cp1251"))
    code, out, err = run(["extract", str(path)], ScriptedClient([]))
    assert code == 1
    assert out == ""
    assert str(path) in err


def test_validation_failure_exits_2_with_empty_result():
    code, out, _ = run(["extract", str(DOC_02)], ScriptedClient(["{", "{"]))
    assert code == 2
    data = json.loads(out)
    assert data["doc_type"] == "unknown"
    assert any(w.startswith("validation_failed:") for w in data["warnings"])


def test_ollama_unavailable_exits_3_with_hint():
    code, out, err = run(["extract", str(DOC_02)], UnavailableClient())
    assert code == 3
    assert out == ""
    assert "ollama serve" in err
    assert "ollama pull" in err


def test_empty_document_exits_0():
    code, out, _ = run(["extract", "-"], ScriptedClient([]), stdin_text="")
    assert code == 0
    assert "empty_document" in json.loads(out)["warnings"]


# Крайние случаи из тикета

def test_no_arguments_exits_1_with_usage():
    code, out, err = run([], ScriptedClient([]))
    assert code == 1
    assert out == ""
    assert "usage" in err.lower()


def test_unknown_subcommand_exits_1():
    code, out, err = run(["frobnicate"], ScriptedClient([]))
    assert code == 1
    assert out == ""
    assert "frobnicate" in err


def test_bom_utf8_file_is_read(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbf" + DOC_02.read_bytes())
    client = ScriptedClient([VALID_JSON])
    code, _, _ = run(["extract", str(path)], client)
    assert code == 0
    assert "﻿" not in client.calls[0][-1]["content"]


def test_cyrillic_is_not_escaped_in_stdout():
    code, out, _ = run(["extract", str(DOC_02)], ScriptedClient([VALID_JSON]))
    assert code == 0
    assert "Разработка сайта" in out
    assert "\\u" not in out
