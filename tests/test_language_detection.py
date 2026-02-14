from server.assistant.language import detect_response_language


def test_detects_german_from_umlaut() -> None:
    assert detect_response_language("Grüße aus Köln", fallback="en") == "de"


def test_detects_german_from_function_words() -> None:
    assert detect_response_language("ich danke dir", fallback="en") == "de"


def test_uses_fallback_when_no_markers_present() -> None:
    assert detect_response_language("ok", fallback="de") == "de"


def test_invalid_fallback_defaults_to_english() -> None:
    assert detect_response_language("ok", fallback="fr") == "en"
