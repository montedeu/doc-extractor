import json

import pytest

import eval as eval_script
from doc_extractor.llm import LLMUnavailableError

VALID = {
    "doc_type": "invoice", "supplier_name": None, "supplier_bin": None,
    "doc_number": "1", "doc_date": None, "currency": None,
    "total_amount": None, "vat_amount": None, "line_items": [],
}
GOLD = {**VALID, "expect_warnings": False}


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name in ("docs", "gold", "responses"):
        (tmp_path / name).mkdir()
    for doc in ("doc_01", "doc_02"):
        (tmp_path / "docs" / f"{doc}.txt").write_text("текст документа", encoding="utf-8")
        (tmp_path / "gold" / f"{doc}.json").write_text(json.dumps(GOLD), encoding="utf-8")
    monkeypatch.setattr(eval_script, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(eval_script, "GOLD_DIR", tmp_path / "gold")
    monkeypatch.setattr(eval_script, "RESPONSES_DIR", tmp_path / "responses")
    monkeypatch.setattr(eval_script, "REPORT_PATH", tmp_path / "out" / "eval_report.json")
    return tmp_path


def write_response(env, doc, responses, model="m:1"):
    d = eval_script.model_dir(model)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc}.json").write_text(json.dumps({"responses": responses}), encoding="utf-8")


def run(*args):
    return eval_script.main(["--model", "m:1", *args])


def test_replay_missing_model_dir_returns_1(env, capsys):
    assert run() == 1
    assert "make record" in capsys.readouterr().err


def test_replay_ok_returns_0(env):
    for doc in ("doc_01", "doc_02"):
        write_response(env, doc, [json.dumps(VALID)])
    assert run() == 0
    assert (env / "out" / "eval_report.json").exists()


def test_replay_missing_response_is_reported_and_returns_1(env, capsys):
    write_response(env, "doc_01", [json.dumps(VALID)])
    write_response(env, "doc_02", ["{", "{"][:1])  # на повтор не хватает ответа
    assert run() == 1
    out = capsys.readouterr().out
    assert "doc_02: ОШИБКА" in out and "make record" in out
    assert "doc_01: " in out  # отчёт напечатан


def test_replay_unused_responses_is_error(env, capsys):
    write_response(env, "doc_01", [json.dumps(VALID), "лишний"])
    write_response(env, "doc_02", [json.dumps(VALID)])
    assert run() == 1
    assert "doc_01: ОШИБКА" in capsys.readouterr().out


def test_replay_broken_file_is_error_not_crash(env, capsys):
    write_response(env, "doc_01", [json.dumps(VALID)])
    d = eval_script.model_dir("m:1")
    (d / "doc_02.json").write_text("{не json", encoding="utf-8")
    assert run() == 1
    assert "doc_02: ОШИБКА" in capsys.readouterr().out
    (d / "doc_02.json").write_text(json.dumps({"model": "x"}), encoding="utf-8")
    assert run() == 1


def test_model_dir_sanitizes_name(env):
    assert eval_script.model_dir("a/b:c d").name == "a_b_c_d"


def test_record_unavailable_on_second_doc_writes_nothing(env, monkeypatch):
    class Flaky:
        calls = 0

        def __init__(self, model=None):
            self.model = model

        def complete(self, messages):
            Flaky.calls += 1
            if Flaky.calls == 2:
                raise LLMUnavailableError("нет Ollama")
            return json.dumps(VALID)

    monkeypatch.setattr(eval_script, "OllamaClient", Flaky)
    assert run("--mode", "record") == 3
    assert list((env / "responses").rglob("*.json")) == []
