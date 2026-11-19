import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
    stdin = stdin_text if not isinstance(stdin_text, str) else io.StringIO(stdin_text)
    code = main(argv, client=client, stdin=stdin, stdout=out, stderr=err)
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


def test_non_utf8_stdin_bytes_exits_1():
    raw = io.TextIOWrapper(io.BytesIO("Счёт на оплату".encode("cp1251")))
    out, err = io.StringIO(), io.StringIO()
    code = main(["extract", "-"], client=ScriptedClient([]), stdin=raw,
                stdout=out, stderr=err)
    assert code == 1
    assert out.getvalue() == ""
    assert "UTF-8" in err.getvalue()


def test_nul_byte_in_path_exits_1():
    code, out, err = run(["extract", "bad\x00name.txt"], ScriptedClient([]))
    assert code == 1
    assert out == ""
    assert err != ""


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
    assert "connection refused" in err
    assert "ollama serve" in err
    assert "ollama pull" in err
    assert err.index("connection refused") < err.index("ollama serve")


def test_empty_document_exits_0():
    code, out, _ = run(["extract", "-"], ScriptedClient([]), stdin_text="")
    assert code == 0
    assert "empty_document" in json.loads(out)["warnings"]


def test_help_prints_usage_to_stdout_and_exits_0():
    for flag in ("-h", "--help"):
        code, out, err = run([flag], ScriptedClient([]))
        assert code == 0
        assert "usage" in out.lower()
        assert err == ""


def test_stdout_reconfigured_to_utf8_when_not_passed(monkeypatch):
    class FakeStdout(io.StringIO):
        encoding_set = None

        def reconfigure(self, encoding):
            self.encoding_set = encoding

    fake = FakeStdout()
    monkeypatch.setattr(sys, "stdout", fake)
    code = main(["extract", "-"], client=ScriptedClient([]),
                stdin=io.StringIO(""), stderr=io.StringIO())
    assert code == 0
    assert fake.encoding_set == "utf-8"
    assert "empty_document" in json.loads(fake.getvalue())["warnings"]


def test_subprocess_empty_stdin_exits_0():
    # проверяем код текущего checkout, а не установленный в .venv пакет
    proc = subprocess.run(
        [sys.executable, "-m", "doc_extractor", "extract", "-"],
        input=b"", capture_output=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT,
    )
    assert proc.returncode == 0
    data = json.loads(proc.stdout.decode("utf-8"))
    assert "empty_document" in data["warnings"]


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


def test_crlf_stdin_equals_file(tmp_path):
    data = DOC_02.read_bytes().replace(b"\n", b"\r\n")
    path = tmp_path / "crlf.txt"
    path.write_bytes(data)
    file_client = ScriptedClient([VALID_JSON])
    assert run(["extract", str(path)], file_client)[0] == 0
    stdin_client = ScriptedClient([VALID_JSON])
    stdin = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8")
    assert run(["extract", "-"], stdin_client, stdin_text=stdin)[0] == 0
    from_file = file_client.calls[0][-1]["content"]
    assert stdin_client.calls[0][-1]["content"] == from_file
    assert "\r" not in from_file


@pytest.mark.parametrize("raw, expected", [
    (b"\xef\xbb\xbfone\r\ntwo\rthree\n", "one\ntwo\nthree\n"),
    (b"\xef\xbb\xbf\r\nx", "\nx"),
])
def test_stdin_bom_and_newlines_normalized(raw, expected):
    client = ScriptedClient([VALID_JSON])
    stdin = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8")
    assert run(["extract", "-"], client, stdin_text=stdin)[0] == 0
    assert client.calls[0][-1]["content"] == expected


def test_cyrillic_is_not_escaped_in_stdout():
    code, out, _ = run(["extract", str(DOC_02)], ScriptedClient([VALID_JSON]))
    assert code == 0
    assert "Разработка сайта" in out
    assert "\\u" not in out
