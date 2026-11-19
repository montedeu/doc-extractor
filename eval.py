"""Прогон тестового набора через пайплайн и отчёт по эталонам.

python eval.py [--mode replay|live|record] [--model M]
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from doc_extractor.llm import (
    LLMClient,
    LLMUnavailableError,
    OllamaClient,
    ScriptedClient,
    ScriptExhaustedError,
    model_from_env,
)
from doc_extractor.pipeline import extract
from doc_extractor import prompt as prompt_module
from doc_extractor.scoring import evaluate, format_report

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "data" / "docs"
GOLD_DIR = ROOT / "data" / "gold"
RESPONSES_DIR = ROOT / "results" / "responses"
REPORT_PATH = ROOT / "results" / "eval_report.json"

EXIT_NO_RESPONSES = 1
EXIT_LLM_UNAVAILABLE = 3

# версия промпта: хеш всего модуля prompt.py (системный промпт и шаблон повтора)
PROMPT_SHA = hashlib.sha256(Path(prompt_module.__file__).read_bytes()).hexdigest()
PROMPT_MISSING_WARNING = "записанные ответы без отметки версии промпта — актуальность не проверить"
PROMPT_WARNING = "записанные ответы сняты на другой версии промпта — make record"


class RecordingClient:
    """Вызывает настоящий клиент и сохраняет каждый возвращённый текст."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.responses: list[str] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        raw = self.inner.complete(messages)
        self.responses.append(raw)
        return raw


def model_dir(model: str) -> Path:
    return RESPONSES_DIR / re.sub(r"[^A-Za-z0-9._-]", "_", model)


def load_responses(path: Path) -> tuple[list[str], str | None]:
    """Читает записанные ответы и sha256 промпта (None, если поля нет).

    ValueError, если файл битый или без ключа responses.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        responses = data["responses"]
    except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as e:
        raise ValueError(f"битый файл {path.name} ({type(e).__name__}); запишите: make record") from e
    if not isinstance(responses, list) or not all(isinstance(r, str) for r in responses):
        raise ValueError(f"в {path.name} responses не список строк; запишите: make record")
    return responses, data.get("prompt_sha256")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["replay", "live", "record"], default="replay")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)
    model = args.model or model_from_env()

    doc_ids = sorted(p.stem for p in DOCS_DIR.glob("*.txt"))
    golds: dict[str, dict] = {}
    for doc_id in doc_ids:
        gold_path = GOLD_DIR / f"{doc_id}.json"
        if gold_path.exists():
            golds[doc_id] = json.loads(gold_path.read_text(encoding="utf-8"))
        else:
            print(f"Нет эталона для {doc_id}, документ пропущен", file=sys.stderr)

    if args.mode == "replay" and not any(model_dir(model).glob("*.json")):
        print(
            f"Нет записанных ответов для модели {model} ({model_dir(model)}).\n"
            f"Запишите их: make record",
            file=sys.stderr,
        )
        return EXIT_NO_RESPONSES

    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    recorded: dict[str, list[str]] = {}
    prompt_missing = False
    prompt_stale = False
    for doc_id in golds:
        text = (DOCS_DIR / f"{doc_id}.txt").read_text(encoding="utf-8-sig")
        recorder: RecordingClient | None = None
        if args.mode == "replay":
            path = model_dir(model) / f"{doc_id}.json"
            if not path.exists():
                errors[doc_id] = f"нет записанного ответа {path.name}; запишите: make record"
                continue
            try:
                responses, recorded_sha = load_responses(path)
            except ValueError as e:
                errors[doc_id] = str(e)
                continue
            if recorded_sha is None:
                prompt_missing = True
            elif recorded_sha != PROMPT_SHA:
                prompt_stale = True
            client: LLMClient = ScriptedClient(responses)
        else:
            client = OllamaClient(model)
            if args.mode == "record":
                client = recorder = RecordingClient(client)
        try:
            extraction = extract(text, client)
        except ScriptExhaustedError:
            errors[doc_id] = (
                "ошибка воспроизведения: пайплайн просит больше ответов, чем записано; "
                "перезапишите: make record"
            )
            continue
        except LLMUnavailableError as e:
            print(f"{e}\nЗапустите: ollama serve && ollama pull {model}", file=sys.stderr)
            return EXIT_LLM_UNAVAILABLE
        if args.mode == "replay" and client.remaining:
            errors[doc_id] = (
                f"ошибка воспроизведения: осталось {client.remaining} неиспользованных "
                "записанных ответов; перезапишите: make record"
            )
            continue
        results[doc_id] = extraction.result.model_dump(mode="json")
        if recorder is not None:
            recorded[doc_id] = recorder.responses

    if args.mode == "record":
        # пишем только после полного прогона: при сбое мусор в записанные ответы не попадает
        out = model_dir(model)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for doc_id, responses in recorded.items():
            payload = {
                "model": model,
                "recorded_at": stamp,
                "prompt_sha256": PROMPT_SHA,
                "responses": responses,
            }
            (out / f"{doc_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    if prompt_missing:
        print(PROMPT_MISSING_WARNING, file=sys.stderr)
    if prompt_stale:
        print(PROMPT_WARNING, file=sys.stderr)

    report = evaluate(results, golds)
    for doc_id, message in errors.items():
        report.per_doc[doc_id] = {"error": message}
    print(format_report(report, total_docs=len(golds)))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "model": model,
                "mode": args.mode,
                "outcomes": {f: dict(c) for f, c in report.outcomes.items()},
                "accuracy": {f: report.accuracy(f) for f in report.outcomes},
                "order_ok": report.order_ok,
                "order_total": report.order_total,
                "warnings_ok": report.warnings_ok,
                "warnings_total": report.warnings_total,
                "per_doc": report.per_doc,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if errors:
        print(f"Ошибок воспроизведения: {len(errors)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
