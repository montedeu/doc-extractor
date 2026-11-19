import hashlib
import json

import pytest

import eval as eval_script
from doc_extractor.llm import LLMUnavailableError, ScriptExhaustedError
from doc_extractor.prompt import SYSTEM_PROMPT

PROMPT_SHA = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
PROMPT_WARNING = "записанные ответы сняты на другой версии промпта — make record"

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


def write_payload(doc, payload, model="m:1"):
    d = eval_script.model_dir(model)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc}.json").write_text(json.dumps(payload), encoding="utf-8")


def write_response(env, doc, responses, model="m:1"):
    write_payload(doc, {"responses": responses, "prompt_sha256": PROMPT_SHA}, model)


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


def test_script_exhausted_is_assertion_error():
    assert issubclass(ScriptExhaustedError, AssertionError)


def test_report_shows_documents_count(env, capsys):
    (env / "docs" / "doc_03.txt").write_text("текст", encoding="utf-8")
    (env / "gold" / "doc_03.json").write_text(json.dumps(GOLD), encoding="utf-8")
    write_response(env, "doc_01", [json.dumps(VALID)])
    write_response(env, "doc_02", [json.dumps(VALID)])
    # doc_03 без записанного ответа: ошибка воспроизведения
    assert run() == 1
    assert "Документов в расчёте: 2 из 3" in capsys.readouterr().out


def test_record_writes_prompt_sha256(env, monkeypatch):
    class Ok:
        def __init__(self, model=None):
            self.model = model

        def complete(self, messages):
            return json.dumps(VALID)

    monkeypatch.setattr(eval_script, "OllamaClient", Ok)
    assert run("--mode", "record") == 0
    path = eval_script.model_dir("m:1") / "doc_01.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["prompt_sha256"] == hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def test_replay_warns_on_prompt_mismatch_but_exits_0(env, capsys):
    for doc in ("doc_01", "doc_02"):
        write_payload(doc, {"responses": [json.dumps(VALID)], "prompt_sha256": "0" * 64})
    assert run() == 0
    err = capsys.readouterr().err
    assert err.count(PROMPT_WARNING) == 1


def test_replay_warns_when_prompt_sha_missing(env, capsys):
    for doc in ("doc_01", "doc_02"):
        write_payload(doc, {"responses": [json.dumps(VALID)]})
    assert run() == 0
    assert PROMPT_WARNING in capsys.readouterr().err


def test_replay_no_warning_on_matching_prompt(env, capsys):
    for doc in ("doc_01", "doc_02"):
        write_response(env, doc, [json.dumps(VALID)])
    assert run() == 0
    assert PROMPT_WARNING not in capsys.readouterr().err
