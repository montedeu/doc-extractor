import json
import sys
from pathlib import Path
from typing import TextIO

from doc_extractor.llm import LLMClient, LLMUnavailableError, OllamaClient, model_from_env
from doc_extractor.pipeline import extract

USAGE = "usage: python -m doc_extractor extract <путь | ->"

EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_VALIDATION_FAILED = 2
EXIT_LLM_UNAVAILABLE = 3


def main(
    argv: list[str] | None = None,
    client: LLMClient | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    args = sys.argv[1:] if argv is None else argv
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr

    if not args:
        print(USAGE, file=stderr)
        return EXIT_INPUT_ERROR
    if args[0] != "extract":
        print(f"Неизвестная подкоманда: {args[0]}\n{USAGE}", file=stderr)
        return EXIT_INPUT_ERROR
    if len(args) != 2:
        print(USAGE, file=stderr)
        return EXIT_INPUT_ERROR

    source = args[1]
    if source == "-":
        text = stdin.read()
    else:
        try:
            # utf-8-sig снимает BOM, если он есть
            text = Path(source).read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            print(f"Файл не в кодировке UTF-8: {source}", file=stderr)
            return EXIT_INPUT_ERROR
        except OSError as e:
            print(f"Не могу прочитать файл {source}: {e.strerror}", file=stderr)
            return EXIT_INPUT_ERROR

    if client is None:
        client = OllamaClient(model_from_env())
    try:
        extraction = extract(text, client)
    except LLMUnavailableError as e:
        model = getattr(client, "model", None) or model_from_env()
        print(
            f"Ollama недоступна ({e}). "
            f"Запустите: ollama serve && ollama pull {model}",
            file=stderr,
        )
        return EXIT_LLM_UNAVAILABLE

    payload = json.dumps(
        extraction.result.model_dump(mode="json"), ensure_ascii=False, indent=2
    )
    print(payload, file=stdout)
    return EXIT_VALIDATION_FAILED if extraction.validation_failed else EXIT_OK
