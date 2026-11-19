"""Уверенность поля в результате (спецификация #1): по статусам сверки, не по оценке модели."""

from collections.abc import Sequence

from doc_extractor.grounding import FieldStatus
from doc_extractor.schema import ModelAnswer

CONFIDENCE = {"ok": 1.0, "ocr_corrected": 0.7, "warned": 0.5}

_SCALAR_FIELDS = tuple(f for f in ModelAnswer.model_fields if f not in ("doc_type", "line_items"))

# противоречия (conflicts.py) понижают уверенность поля до «warned»
_CONFLICT_FIELDS = {
    "conflicting_totals": "total_amount",
    "amount_words_mismatch": "total_amount",
    "conflicting_dates": "doc_date",
}


def _score(status: dict[str, FieldStatus], key: str) -> float:
    # заполненное поле без статуса сверка не трогала: найдено и проверено по определению
    return CONFIDENCE[status[key]] if key in status else 1.0


def field_confidence(
    answer: ModelAnswer, status: dict[str, FieldStatus], warnings: Sequence[str] = ()
) -> dict[str, float]:
    out: dict[str, float] = {}
    if answer.doc_type != "unknown":  # сверке не подлежит, см. README
        out["doc_type"] = 1.0
    for f in _SCALAR_FIELDS:
        if getattr(answer, f) is not None:
            out[f] = _score(status, f)
    if answer.line_items:
        scores = []
        for i, item in enumerate(answer.line_items):
            scores.append(1.0)  # name: сверка оставляет только позиции с найденным названием
            for f in ("qty", "price"):
                key = f"line_items[{i}].{f}"
                if getattr(item, f) is not None:
                    scores.append(_score(status, key))
                elif key in status:  # обнулено сверкой
                    scores.append(CONFIDENCE["warned"])
        out["line_items"] = min(scores)
    for w in warnings:
        field = _CONFLICT_FIELDS.get(w.partition(":")[0])
        if field in out:
            out[field] = min(out[field], CONFIDENCE["warned"])
    return out
