from server.llm.prompt import clean_for_tts


def test_removes_german_source_section_and_links() -> None:
    text = (
        "Hessen hat eine lange Geschichte.\n\n"
        "Quellen:\n"
        "[1] https://example.com/a\n"
        "[2] https://example.com/b"
    )

    cleaned = clean_for_tts(text)

    assert "Quellen" not in cleaned
    assert "https://" not in cleaned
    assert "[1]" not in cleaned
    assert "Hessen hat eine lange Geschichte." in cleaned


def test_removes_cjk_style_and_bracket_citations() -> None:
    text = "Frankfurt ist ein Finanzzentrum【1†source】 in Deutschland [2]."

    cleaned = clean_for_tts(text)

    assert "【" not in cleaned
    assert "】" not in cleaned
    assert "[2]" not in cleaned
    assert "Frankfurt ist ein Finanzzentrum" in cleaned


def test_removes_parenthetical_source_marker() -> None:
    text = "Das ist korrekt (Quelle: Statistisches Bundesamt)."

    cleaned = clean_for_tts(text)

    assert "Quelle:" not in cleaned
    assert cleaned.startswith("Das ist korrekt")
