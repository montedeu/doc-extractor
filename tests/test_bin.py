import json
from pathlib import Path

import pytest

from doc_extractor.bin_check import bin_checksum_ok
from doc_extractor.llm import ScriptedClient
from doc_extractor.normalize import analyze
from doc_extractor.pipeline import extract

DOCS = Path(__file__).resolve().parent.parent / "data/docs"


def _answer(supplier_bin: str) -> str:
    return json.dumps({
        "doc_type": "invoice", "supplier_name": None, "supplier_bin": supplier_bin,
        "doc_number": None, "doc_date": None, "currency": None,
        "total_amount": None, "vat_amount": None, "line_items": [],
    })


def run_text(text: str, supplier_bin: str):
    return extract(text, ScriptedClient([_answer(supplier_bin)])).result


def run(doc: str, supplier_bin: str):
    return run_text((DOCS / f"{doc}.txt").read_text(encoding="utf-8"), supplier_bin)


def test_checksum_known_values():
    for ok in ("210640900180", "150940016484", "190440012341"):
        assert bin_checksum_ok(ok)
    assert not bin_checksum_ok("170340021554")


def test_checksum_rejects_malformed():
    assert not bin_checksum_ok("12345")
    assert not bin_checksum_ok("2106409001a0")


def test_invented_bin_becomes_null():
    r = run("doc_02", "123456789012")
    assert r.supplier_bin is None
    assert any(w.startswith("not_in_text: supplier_bin") for w in r.warnings)


def test_iin_of_individual_entrepreneur_is_not_bin():
    r = run("doc_10", "850612300451")
    assert r.supplier_bin is None
    assert any(w.startswith("bin_not_supplier") for w in r.warnings)


def test_buyer_bin_is_rejected():
    r = run("doc_10", "051240019732")
    assert r.supplier_bin is None
    assert any(w.startswith("bin_not_supplier") for w in r.warnings)
    r = run("doc_05", "161140033925")
    assert r.supplier_bin is None
    assert any(w.startswith("bin_not_supplier") for w in r.warnings)


def test_supplier_bin_in_act_is_kept():
    r = run("doc_05", "200540028712")
    assert r.supplier_bin == "200540028712"
    assert r.warnings == []


@pytest.mark.parametrize("doc,value,raw", [
    ("doc_03", "210640900180", "2l064090Ol80"),
    ("doc_07", "150940016484", "l5O94OOl6484"),
])
def test_ocr_bin_corrected_when_checksum_ok(doc, value, raw):
    r = run(doc, value)
    assert r.supplier_bin == value
    assert f"ocr_corrected: supplier_bin {raw} → {value}" in r.warnings


def test_bad_checksum_kept_with_warning():
    r = run("doc_01", "170340021554")
    assert r.supplier_bin == "170340021554"
    assert r.warnings == ["bin_checksum_failed: supplier_bin=170340021554"]


def test_phone_is_not_bin():
    r = run("doc_09", "770123456780")
    assert r.supplier_bin is None
    assert any(w.startswith("not_in_text: supplier_bin") for w in r.warnings)


def test_iban_gives_no_candidate():
    facts = analyze("ИИК KZ86 601A 8710 0012 3456, БИК HSBKKZKX")
    assert facts.bin_candidates == []


@pytest.mark.parametrize("label", ["БСН/БИН:", "БСН / БИН:"])
def test_bin_labels(label):
    r = run_text(f"ТОО Х\n{label} 210640900180\nСчёт", "210640900180")
    assert r.supplier_bin == "210640900180"
    assert r.warnings == []


def test_nearest_label_decides_in_shared_line():
    text = 'Поставщик: ТОО "А", БИН 210640900180; Покупатель: ТОО "Б", БИН 150940016484'
    assert run_text(text, "150940016484").supplier_bin is None
    assert run_text(text, "210640900180").supplier_bin == "210640900180"


V = "210640900180"
V2 = "190440012341"


def _bn(r):
    return [w for w in r.warnings if w.startswith("bin_not_supplier")]


@pytest.mark.parametrize("label", ["БИН", "бин", "БСН", "BIN", "Bin", "БlН"])
def test_bin_label_variants_keep_value(label):
    r = run_text(f"ТОО Х\n{label} {V}\nСчёт", V)
    assert r.supplier_bin == V
    assert r.warnings == []


def test_combined_label_is_not_iin_only_role_decides():
    for label in ("БИН/ИИН:", "ИИН/БИН:", "БИН (ИИН)", "БИН / ИИН", "БСН/ЖСН", "ИИН/БИН"):
        assert run_text(f"Поставщик: ТОО А, {label} {V}", V).supplier_bin == V
        assert run_text(f"Покупатель: ТОО А, {label} {V}", V).supplier_bin is None
        assert run_text(f"{label} {V}", V).supplier_bin == V


@pytest.mark.parametrize("label", ["ИИН", "IIN", "ІІН", "ЖСН"])
def test_standalone_iin_label_rejects(label):
    r = run_text(f"{label} {V}", V)
    assert r.supplier_bin is None
    assert len(_bn(r)) == 1


def test_any_buyer_occurrence_rejects():
    r = run_text(f"Покупатель: БИН {V}\nпрочее: {V}", V)
    assert r.supplier_bin is None
    assert len(_bn(r)) == 1


@pytest.mark.parametrize("line", [
    f"БИН-{V}", f"БИН:{V}/", f"|{V}|", f"№{V}",
    f'ТОО "Альфа", БИН {V}, г. Алматы',
    f"БИН {V}.",
    f"(БИН {V}); адрес",
    f"БИН:{V}",
])
def test_candidate_allows_punctuation_around(line):
    assert [c.digits for c in analyze(line).bin_candidates] == [V]


@pytest.mark.parametrize("line", [f"БИН {V}5", f"БИН 5{V}", f"БИН {V}а", f"x{V}"])
def test_candidate_rejects_adjacent_chars(line):
    assert analyze(line).bin_candidates == []


def test_best_candidate_by_checksum_and_ocr():
    # то же значение дважды: сначала с OCR (сходится), потом чистое
    text = f"БИН 2l064090Ol80\nБИН {V}"
    r = run_text(text, V)
    assert r.supplier_bin == V
    assert r.warnings == []


def test_ocr_candidate_with_bad_checksum_is_rejected():
    r = run_text("БИН l70340021554", "170340021554")
    assert r.supplier_bin is None
    assert "bin_checksum_failed: supplier_bin=170340021554" in r.warnings


def test_iin_latin_and_receiver_is_not_buyer():
    assert run_text(f"IIN {V}", V).supplier_bin is None
    r = run_text(f"Получатель: ТОО Бета, БИН {V}", V)
    assert r.supplier_bin == V


def test_role_header_in_previous_lines():
    r = run_text(f"Покупатель:\nТОО Альфа\nБИН {V}", V)
    assert r.supplier_bin is None
    assert len(_bn(r)) == 1
    assert run_text(f"Поставщик:\nТОО Альфа\nБИН {V}", V).supplier_bin == V
    # заголовок в пределах абзаца, до 5 строк вверх
    assert run_text(f"Покупатель:\nа\nб\nв\nг\nБИН {V}", V).supplier_bin is None
    assert run_text(f"Покупатель:\nа\nб\nв\nг\nд\nБИН {V}", V).supplier_bin == V
    # пустая строка обрывает абзац
    assert run_text(f"Покупатель:\n\nБИН {V}", V).supplier_bin == V


def test_labels_have_word_boundaries_and_ignore_quoted_names():
    assert run_text(f"бином {V}", V).supplier_bin == V
    r = run_text(f'ТОО "Заказчик-Сервис", БИН {V}', V)
    assert r.supplier_bin == V
    r = run_text(f"Заказчиксервис БИН {V}", V)
    assert r.supplier_bin == V


def test_role_label_on_the_right():
    r = run_text(f"{V} — БИН покупателя", V)
    assert r.supplier_bin is None
    assert len(_bn(r)) == 1


def test_nulled_bin_has_no_status():
    from doc_extractor.grounding import ground
    from doc_extractor.schema import ModelAnswer

    a = ModelAnswer.model_validate_json(_answer(V))
    g = ground(a, analyze(f"Покупатель: БИН {V}"))
    assert g.answer.supplier_bin is None
    assert "supplier_bin" not in g.status
