import json
from pathlib import Path

from doc_extractor.llm import ScriptedClient
from doc_extractor.numerals import find_amounts_in_words, parse_amount_words
from doc_extractor.pipeline import extract

ROOT = Path(__file__).resolve().parent.parent


def text_of(n: str) -> str:
    return (ROOT / f"data/docs/doc_{n}.txt").read_text(encoding="utf-8")


def test_russian_amount():
    assert parse_amount_words("Триста пятьдесят тысяч") == 350000
    assert parse_amount_words("сто двадцать одна тысяча пятьсот") == 121500


def test_kazakh_amount():
    assert parse_amount_words("Үш жүз қырық мың") == 340000
    assert parse_amount_words("Үш жүз алпыс мың") == 360000
    assert parse_amount_words("бір миллион екі жүз мың") == 1200000


def test_find_amounts_in_document():
    assert find_amounts_in_words(text_of("05")) == [350000]
    assert find_amounts_in_words(text_of("03")) == [340000]
    assert find_amounts_in_words(text_of("08")) == [360000]


def test_no_amount_words_in_plain_text():
    assert find_amounts_in_words(text_of("01")) == []


def test_total_only_in_words_is_grounded():
    path = ROOT / "data/gold/doc_08.json"
    gold = json.loads(path.read_text(encoding="utf-8"))
    gold = {k: v for k, v in gold.items() if not k.startswith("_") and k != "expect_warnings"}
    gold["total_amount"] = 360000
    answer = json.dumps(gold, ensure_ascii=False)
    r = extract(text_of("08"), ScriptedClient([answer])).result
    assert r.total_amount == 360000
    assert not any("total_amount" in w for w in r.warnings)


def test_edge_cases():
    assert parse_amount_words("тысяча") == 1000
    assert parse_amount_words("одна тысяча") == 1000
    assert parse_amount_words("двести два") == 202
    assert parse_amount_words("семьдесят семь") == 77
    assert parse_amount_words("не число") is None


def test_context_filter():
    for t in ["Он сказал", "Один раз в одно место", "СТО Авто", "один из вариантов",
              "12 тысяч рублей"]:
        assert find_amounts_in_words(t) == [], t
    assert find_amounts_in_words("пять тысяч рублей пятьдесят копеек") == [5000]
    assert find_amounts_in_words("Итого: сто USD") == [100]


def test_oblique_forms():
    assert find_amounts_in_words("двух тысяч тенге") == [2000]
    assert find_amounts_in_words("одну тысячу тенге") == [1000]
    assert find_amounts_in_words("трёх тысяч тенге") == [3000]
    assert find_amounts_in_words("пятисот тенге") == [500]


def test_latin_homoglyphs():
    assert find_amounts_in_words("пятьсоt тенге") == [500]  # латинская t
    assert find_amounts_in_words("пятьсот тенге".replace("о", "o")) == [500]  # латинская o


def test_line_break_inside_phrase():
    assert find_amounts_in_words("Пятьсот\nтысяч тенге") == [500000]
