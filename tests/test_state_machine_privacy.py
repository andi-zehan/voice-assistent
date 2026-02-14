from assistant.telemetry import llm_metrics_payload, stt_metrics_payload


def test_stt_metrics_hide_text_by_default() -> None:
    result = {
        "text": "sensitive speech",
        "language": "en",
        "duration_s": 1.2,
        "transcription_time_s": 0.4,
        "avg_logprob": -0.2,
        "no_speech_prob": 0.01,
    }

    payload = stt_metrics_payload(result, include_text=False)

    assert "text" not in payload
    assert payload["text_chars"] == len(result["text"])


def test_stt_metrics_can_include_text_when_enabled() -> None:
    result = {"text": "debug transcript"}

    payload = stt_metrics_payload(result, include_text=True)

    assert payload["text"] == "debug transcript"


def test_llm_metrics_hide_text_by_default() -> None:
    result = {
        "text": "sensitive answer",
        "model": "openai/gpt-5-chat",
        "elapsed_s": 1.4,
        "ttft_s": 0.7,
    }

    payload = llm_metrics_payload(result, include_text=False)

    assert "text" not in payload
    assert payload["text_chars"] == len(result["text"])


def test_llm_metrics_can_include_text_when_enabled() -> None:
    result = {"text": "debug answer"}

    payload = llm_metrics_payload(result, include_text=True)

    assert payload["text"] == "debug answer"
