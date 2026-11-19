import os
from typing import Protocol

from doc_extractor.schema import model_answer_json_schema

DEFAULT_MODEL = "qwen3.5:9b"
NUM_CTX = 8192
TIMEOUT_SECONDS = 300


def model_from_env() -> str:
    return os.environ.get("DOC_EXTRACTOR_MODEL") or DEFAULT_MODEL


class LLMUnavailableError(Exception):
    """Ollama не запущена или модель не скачана."""


class ScriptExhaustedError(AssertionError):
    """ScriptedClient: заготовленные ответы кончились, а вызов пришёл."""


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
            raise ScriptExhaustedError(
                f"ScriptedClient: ответы кончились, вызов №{len(self.calls)} лишний"
            )
        return self._responses.pop(0)

    @property
    def remaining(self) -> int:
        return len(self._responses)


class OllamaClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or model_from_env()

    def complete(self, messages: list[dict[str, str]]) -> str:
        import httpx
        import ollama

        try:
            client = ollama.Client(timeout=TIMEOUT_SECONDS)
            response = client.chat(
                model=self.model,
                messages=messages,
                think=False,
                options={"temperature": 0, "num_ctx": NUM_CTX},
                format=model_answer_json_schema(),
            )
        except ollama.ResponseError as e:
            if e.status_code == 404 or "not found" in str(e).lower():
                raise LLMUnavailableError(
                    f"модель {self.model} не скачана: {e}"
                ) from e
            raise LLMUnavailableError(f"Ollama вернула ошибку: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMUnavailableError(
                f"Ollama не ответила за {TIMEOUT_SECONDS} с: {e}"
            ) from e
        except (httpx.HTTPError, ConnectionError) as e:
            raise LLMUnavailableError(f"Ollama недоступна: {e}") from e
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError):
            return ""
        return content or ""
