from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocType = Literal["invoice", "commercial_offer", "act", "unknown"]
Currency = Literal["KZT", "USD", "EUR"]


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    qty: float | None = Field(ge=0, allow_inf_nan=False)
    price: float | None = Field(ge=0, allow_inf_nan=False)


class ModelAnswer(BaseModel):
    """Ответ модели: без field_confidence и warnings (ADR 0003)."""

    model_config = ConfigDict(extra="forbid")

    doc_type: DocType
    supplier_name: str | None
    supplier_bin: str | None = Field(pattern=r"^[0-9]{12}$")
    doc_number: str | None
    doc_date: date | None
    currency: Currency | None
    total_amount: float | None = Field(ge=0, allow_inf_nan=False)
    vat_amount: float | None = Field(ge=0, allow_inf_nan=False)
    line_items: list[LineItem]


class ExtractionResult(ModelAnswer):
    field_confidence: dict[str, float]
    warnings: list[str]


def empty_result(warnings: list[str]) -> ExtractionResult:
    return ExtractionResult(
        doc_type="unknown",
        supplier_name=None,
        supplier_bin=None,
        doc_number=None,
        doc_date=None,
        currency=None,
        total_amount=None,
        vat_amount=None,
        line_items=[],
        field_confidence={},
        warnings=list(warnings),
    )


def model_answer_json_schema() -> dict:
    return ModelAnswer.model_json_schema()
