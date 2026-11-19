"""Уверенность поля в результате (спецификация #1): по статусам сверки, не по оценке модели."""

from doc_extractor.grounding import FieldStatus
from doc_extractor.schema import ModelAnswer

CONFIDENCE = {"ok": 1.0, "ocr_corrected": 0.7, "warned": 0.5}

_SCALAR_FIELDS = (
    "supplier_name", "supplier_bin", "doc_number", "doc_date",
    "currency", "total_amount", "vat_amount",
)


def _score(status: dict[str, FieldStatus], key: str) -> float:
    # заполненное поле без статуса сверка не трогала: найдено и проверено по определению
    return CONFIDENCE[status[key]] if key in status else 1.0


def field_confidence(answer: ModelAnswer, status: dict[str, FieldStatus]) -> dict[str, float]:
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
                if getattr(item, f) is not None:
                    scores.append(_score(status, f"line_items[{i}].{f}"))
        out["line_items"] = float(min(scores))
    return out
