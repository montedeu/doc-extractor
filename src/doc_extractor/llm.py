import os
from typing import Protocol

from doc_extractor.schema import model_answer_json_schema

DEFAULT_MODEL = "qwen3.5:9b"


def model_from_env() -> str:
    return os.environ.get("DOC_EXTRACTOR_MODEL") or DEFAULT_MODEL


class LLMUnavailableError(Exception):
    """Ollama не запущена или модель не скачана."""


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class ScriptedClient:
    """Отдаёт заранее заданные ответы по очереди; для тестов."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError(
                f"ScriptedClient: ответы кончились, вызов №{len(self.calls)} лишний"
            )
        return self._responses.pop(0)


class OllamaClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or model_from_env()

    def complete(self, messages: list[dict[str, str]]) -> str:
        import ollama

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                think=False,
                options={"temperature": 0},
                format=model_answer_json_schema(),
            )
        except ConnectionError as e:
            raise LLMUnavailableError(f"Ollama не запущена: {e}") from e
        except ollama.ResponseError as e:
            if e.status_code == 404 or "not found" in str(e).lower():
                raise LLMUnavailableError(
                    f"модель {self.model} не скачана: {e}"
                ) from e
            raise
        return response["message"]["content"]
