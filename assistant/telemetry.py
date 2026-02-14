"""Helpers for privacy-aware metrics payloads."""


def stt_metrics_payload(stt_result: dict, include_text: bool = False) -> dict:
    """Build STT metrics payload with optional transcript text."""
    payload = {
        "language": stt_result.get("language"),
        "duration_s": stt_result.get("duration_s"),
        "transcription_time_s": stt_result.get("transcription_time_s"),
        "avg_logprob": stt_result.get("avg_logprob"),
        "no_speech_prob": stt_result.get("no_speech_prob"),
        "text_chars": len(stt_result.get("text", "")),
    }
    if include_text:
        payload["text"] = stt_result.get("text", "")
    return payload


def llm_metrics_payload(llm_result: dict, include_text: bool = False) -> dict:
    """Build LLM metrics payload with optional response text."""
    payload = {
        "model": llm_result.get("model"),
        "elapsed_s": llm_result.get("elapsed_s"),
        "ttft_s": llm_result.get("ttft_s"),
        "text_chars": len(llm_result.get("text", "")),
    }
    if include_text:
        payload["text"] = llm_result.get("text", "")
    return payload
