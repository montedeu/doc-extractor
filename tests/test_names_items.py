import json
from pathlib import Path

from doc_extractor.llm import ScriptedClient
from doc_extractor.normalize import name_core
from doc_extractor.pipeline import extract

ROOT = Path(__file__).resolve().parent.parent


def text_of(n: str) -> str:
    return (ROOT / f"data/docs/doc_{n}.txt").read_text(encoding="utf-8")


def gold(n: str) -> dict:
    d = json.loads((ROOT / f"data/gold/doc_{n}.json").read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_") and k != "expect_warnings"}


def run(text: str, answer: dict):
    return extract(text, ScriptedClient([json.dumps(answer, ensure_ascii=False)])).result


def items(*names: str, qty=None, price=None) -> list[dict]:
    return [{"name": n, "qty": qty, "price": price} for n in names]


def test_supplier_name_with_homoglyphs_is_grounded():
    r = run(text_of("07"), gold("07") | {"supplier_name": 'ТОО "Береке Принт"'})
    assert r.supplier_name == 'ТОО "Береке Принт"'
    assert not any("supplier_name" in w for w in r.warnings)


def test_latin_too_is_grounded():
    r = run(text_of("04"), gold("04") | {"supplier_name": 'ТОО "Steppe Cloud Services"'})
    assert r.supplier_name == 'ТОО "Steppe Cloud Services"'


def test_invented_supplier_name_becomes_null():
    r = run(text_of("09"), gold("09") | {"supplier_name": 'ТОО "Данияр Трейд"'})
    assert r.supplier_name is None
    assert any(w.startswith("not_in_text: supplier_name=") for w in r.warnings)


def test_invented_line_item_is_dropped_order_kept():
    g = gold("01")
    a, c = g["line_items"]
    inv = {"name": "Доставка", "qty": 1, "price": 5000}
    r = run(text_of("01"), g | {"line_items": [a, inv, c]})
    assert [i.name for i in r.line_items] == [a["name"], c["name"]]
    assert "line_item_dropped: Доставка" in r.warnings


def test_totals_row_as_item_is_dropped():
    g = gold("01")
    row = {"name": "Итого", "qty": None, "price": 160500}
    r = run(text_of("01"), g | {"line_items": g["line_items"] + [row]})
    assert [i.name for i in r.line_items] == [i["name"] for i in g["line_items"]]
    assert "line_item_dropped: Итого" in r.warnings


def test_bilingual_supplier_russian_variant_grounded():
    r = run(text_of("03"), gold("03") | {"supplier_name": 'ТОО "Кокжиек Курылыс"'})
    assert r.supplier_name == 'ТОО "Кокжиек Курылыс"'


def test_ocr_item_name_grounded():
    g = gold("07")
    r = run(text_of("07"), g | {"line_items": [
        {"name": "Визитки 90x50 мм", "qty": 1000, "price": 25},
        {"name": "Баннер 3x6 м", "qty": 2, "price": 4500},
    ]})
    assert [i.name for i in r.line_items] == ["Визитки 90x50 мм", "Баннер 3x6 м"]
    assert not any("line_item_dropped" in w for w in r.warnings)


def test_opf_only_supplier_becomes_null():
    r = run(text_of("01"), gold("01") | {"supplier_name": "ТОО"})
    assert r.supplier_name is None
    assert any(w.startswith("not_in_text: supplier_name=") for w in r.warnings)


def test_bilingual_item_found_by_parts():
    text = "Счёт\nАтауы / Наименование: Тұқым (бидай) / Семена пшеницы\n"
    r = run(text, gold("08") | {"line_items": items("Тұқым (бидай) / Семена пшеницы")})
    assert len(r.line_items) == 1


def test_bilingual_item_with_invented_part_dropped():
    r = run(text_of("08"), gold("08") | {"line_items": items("Тұқым (бидай) / Семена овса")})
    assert r.line_items == []
    assert any(w.startswith("line_item_dropped") for w in r.warnings)


def test_item_warning_indexes_follow_original_list():
    g = gold("01")
    a, c = g["line_items"]
    c = dict(c, qty=7)  # 7 нет в тексте
    r = run(text_of("01"), g | {"line_items": [items("Доставка")[0], a, c]})
    assert "not_in_text: line_items[2].qty=7" in r.warnings
    assert r.line_items[1].qty is None


def test_name_core():
    assert name_core('T0O "Бepeкe Пpинт"') == "береке принт"
    assert name_core("ТОО") == ""
    assert name_core("«Алтын Дала Агро»") == "алтын дала агро"
