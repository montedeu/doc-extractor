"""Сравнение результатов с эталонами: исходы полей, порядок позиций, предупреждения."""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

Outcome = Literal["correct", "hallucination", "miss", "wrong"]
SCALAR_FIELDS = [
    "doc_type", "supplier_name", "supplier_bin", "doc_number", "doc_date",
    "currency", "total_amount", "vat_amount",
]
ITEM_FIELDS = ["line_items.name", "line_items.qty", "line_items.price"]
ALL_FIELDS = SCALAR_FIELDS + ITEM_FIELDS

_AMOUNT_FIELDS = {"total_amount", "vat_amount", "line_items.qty", "line_items.price"}
_NAME_FIELDS = {"supplier_name", "line_items.name"}

_QUOTES = str.maketrans("", "", "«»\"“”„‟")
# латинские двойники кириллических букв (после casefold)
_HOMOGLYPHS = str.maketrans(
    {"a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м",
     "o": "о", "p": "р", "t": "т", "x": "х", "y": "у"}
)
_ZERO_BETWEEN_LETTERS = re.compile(r"(?<=[^\W\d_])0(?=[^\W\d_])")


def normalize_name(value: str) -> str:
    s = value.translate(_QUOTES)
    prev = None
    while prev != s:  # «T00» требует двух проходов: lookaround не видит соседний 0
        prev = s
        s = _ZERO_BETWEEN_LETTERS.sub("О", s)
    s = s.casefold().translate(_HOMOGLYPHS)
    return " ".join(s.split())


def _equal(field_name: str, pred, gold) -> bool:
    if field_name in _AMOUNT_FIELDS:
        return round(float(pred), 2) == round(float(gold), 2)
    if field_name in _NAME_FIELDS:
        return normalize_name(str(pred)) == normalize_name(str(gold))
    return str(pred).strip() == str(gold).strip()


def compare_field(field: str, pred, gold) -> Outcome:
    if field == "doc_type":
        return "correct" if pred == gold else "wrong"
    if pred is None and gold is None:
        return "correct"
    if gold is None:
        return "hallucination"
    if pred is None:
        return "miss"
    return "correct" if _equal(field, pred, gold) else "wrong"


@dataclass
class Report:
    outcomes: dict[str, Counter] = field(default_factory=dict)
    order_ok: int = 0
    order_total: int = 0
    warnings_ok: int = 0
    warnings_total: int = 0
    per_doc: dict[str, dict] = field(default_factory=dict)

    def accuracy(self, field: str) -> float:
        counter = self.outcomes.get(field, Counter())
        total = sum(counter.values())
        return counter["correct"] / total if total else 0.0


def _match_items(pred_items: list[dict], gold_items: list[dict]):
    """Пары (эталон, позиция модели, индекс позиции модели) по нормализованному наименованию."""
    pool: dict[str, list[tuple[int, dict]]] = {}
    for i, item in enumerate(pred_items):
        pool.setdefault(normalize_name(item["name"]), []).append((i, item))
    pairs, missing = [], []
    for g in gold_items:
        candidates = pool.get(normalize_name(g["name"]))
        if candidates:
            i, p = candidates.pop(0)
            pairs.append((g, p, i))
        else:
            missing.append(g)
    extra = [p for rest in pool.values() for _, p in rest]
    return pairs, missing, extra


def evaluate(results: dict[str, dict], golds: dict[str, dict]) -> Report:
    """Ключ — id документа. Документы без результата или без эталона пропускаются."""
    report = Report(outcomes={f: Counter() for f in ALL_FIELDS})

    for doc, gold in golds.items():
        if doc not in results:
            continue
        res = results[doc]
        issues: list[str] = []

        def add(field: str, outcome: Outcome, detail: str = "") -> None:
            report.outcomes[field][outcome] += 1
            if outcome != "correct":
                issues.append(f"{field}: {outcome}{detail}")

        for f in SCALAR_FIELDS:
            pred, g = res.get(f), gold.get(f)
            add(f, compare_field(f, pred, g), f" (модель {pred!r}, эталон {g!r})")

        pairs, missing, extra = _match_items(
            res.get("line_items") or [], gold.get("line_items") or []
        )
        for g, p, _ in pairs:
            add("line_items.name", "correct")
            for sub in ("qty", "price"):
                f = f"line_items.{sub}"
                add(f, compare_field(f, p.get(sub), g.get(sub)),
                    f" [{g['name']}] (модель {p.get(sub)!r}, эталон {g.get(sub)!r})")
        for g in missing:
            for f in ITEM_FIELDS:
                add(f, "miss", f" [{g['name']}]")
        for p in extra:
            add("line_items.name", "hallucination", f" [{p['name']}]")

        positions = [i for _, _, i in pairs]  # пары идут в порядке эталона
        order_ok = positions == sorted(positions)
        report.order_total += 1
        report.order_ok += order_ok

        warnings_ok = bool(gold.get("expect_warnings")) == bool(res.get("warnings"))
        report.warnings_total += 1
        report.warnings_ok += warnings_ok

        report.per_doc[doc] = {
            "issues": issues,
            "order_ok": order_ok,
            "warnings_ok": warnings_ok,
        }
    return report


def format_report(report: Report) -> str:
    rows = []
    totals: Counter = Counter()
    for f in ALL_FIELDS:
        c = report.outcomes.get(f, Counter())
        totals.update(c)
        rows.append((f, c["correct"], c["hallucination"], c["miss"], c["wrong"], report.accuracy(f)))
    n = sum(totals.values())
    total_acc = totals["correct"] / n if n else 0.0
    total_row = ("ИТОГО", totals["correct"], totals["hallucination"], totals["miss"],
                 totals["wrong"], total_acc)

    width = max(len(r[0]) for r in rows)
    header = (f"{'поле':<{width}}  {'верно':>6}  {'галлюцинации':>12}  {'пропуски':>8}  "
              f"{'неверно':>7}  {'точность':>8}")
    lines = [header, "-" * len(header)]
    for name, ok, h, m, w, acc in rows:
        lines.append(f"{name:<{width}}  {ok:>6}  {h:>12}  {m:>8}  {w:>7}  {acc:>8.0%}")
    lines.append("-" * len(header))
    name, ok, h, m, w, acc = total_row
    lines.append(f"{name:<{width}}  {ok:>6}  {h:>12}  {m:>8}  {w:>7}  {acc:>8.0%}")
    lines.append("")
    lines.append(f"Порядок позиций совпал: {report.order_ok} из {report.order_total} документов")
    lines.append(
        f"Предупреждения совпали с ожиданием: {report.warnings_ok} из {report.warnings_total} документов"
    )

    lines.append("")
    lines.append("По документам (расхождения):")
    for doc, info in sorted(report.per_doc.items()):
        if info.get("error"):
            lines.append(f"  {doc}: ОШИБКА — {info['error']}")
            continue
        items = list(info["issues"])
        if not info["order_ok"]:
            items.append("порядок позиций не совпал")
        if not info["warnings_ok"]:
            items.append("предупреждения не совпали с ожиданием")
        lines.append(f"  {doc}: " + ("всё верно" if not items else ""))
        lines.extend(f"    - {i}" for i in items)
    return "\n".join(lines)
