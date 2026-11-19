import re
from dataclasses import dataclass

from pydantic import ValidationError

from doc_extractor.grounding import ground
from doc_extractor.llm import LLMClient
from doc_extractor.normalize import analyze
from doc_extractor.prompt import build_messages, retry_messages
from doc_extractor.schema import ExtractionResult, ModelAnswer, empty_result

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)


@dataclass
class Extraction:
    result: ExtractionResult
    validation_failed: bool


def _strip_fence(raw: str) -> str:
    raw = raw.strip()
    m = _FENCE.match(raw)
    return m.group(1).strip() if m else raw


def _validate(raw: str) -> tuple[ModelAnswer | None, str]:
    try:
        return ModelAnswer.model_validate_json(_strip_fence(raw)), ""
    except ValidationError as e:
        return None, str(e)


def extract(text: str, client: LLMClient) -> Extraction:
    if not text.strip():
        return Extraction(empty_result(["empty_document"]), False)

    messages = build_messages(text)
    raw = client.complete(messages)
    answer, error = _validate(raw)
    if answer is None:
        raw = client.complete(retry_messages(messages, raw, error))
        answer, error = _validate(raw)
    if answer is None:
        short = " ".join(error.split())[:300]
        return Extraction(empty_result([f"validation_failed: {short}"]), True)

    grounded = ground(answer, analyze(text))
    result = ExtractionResult(
        **grounded.answer.model_dump(),
        field_confidence={},
        warnings=grounded.warnings,
    )
    return Extraction(result, False)
