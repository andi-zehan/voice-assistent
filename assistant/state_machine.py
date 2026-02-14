"""Central state machine orchestrating the voice assistant pipeline."""

import enum
import time
import traceback

import numpy as np

# ANSI color codes for terminal output
_DIM = "\033[90m"       # Gray — state transitions, timing stats
_CYAN = "\033[36m"      # Cyan — user speech (STT transcript)
_BOLD = "\033[1m"       # Bold white — LLM response
_YELLOW = "\033[33m"    # Yellow — events (wake, barge-in, follow-up)
_RED = "\033[31m"       # Red — errors, rejections, timeouts
_RST = "\033[0m"        # Reset

from audio.capture import AudioCapture
from audio.playback import AudioPlayer
from audio.vad import VoiceActivityDetector, UtteranceDetector
from audio.earcon import play_earcon, play_named_earcon
from wake.detector import WakeWordDetector
from stt.whisper_stt import WhisperSTT
from llm.openrouter_client import OpenRouterClient
from llm.prompt import get_system_prompt, build_messages, clean_for_tts
from assistant.language import detect_response_language
from assistant.telemetry import stt_metrics_payload, llm_metrics_payload
from tts import TTSEngine
from assistant.session import Session
from assistant.metrics import MetricsLogger


# Common Whisper hallucinations on silence/noise
_HALLUCINATION_PHRASES = {
    # English
    "thank you for watching",
    "thanks for watching",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "see you in the next video",
    "see you next time",
    "bye bye",
    "thank you",
    "thanks for listening",
    "the end",
    "you",
    "i'm sorry",
    # German
    "danke fürs zuschauen",
    "danke für's zuschauen",
    "vielen dank fürs zuschauen",
    "bis zum nächsten mal",
    "tschüss",
    "untertitel von stephanie geiges",
    "untertitel der amara.org-community",
    "untertitel im auftrag des zdf für funk",
}

_ERROR_MESSAGES: dict[str, str] = {
    "en": "Sorry, something went wrong.",
    "de": "Entschuldigung, da ist etwas schiefgelaufen.",
}


def _check_hallucination(
    text: str,
    no_speech_prob: float,
    avg_logprob: float,
    no_speech_threshold: float = 0.6,
    logprob_threshold: float = -1.0,
) -> tuple[bool, str]:
    """Return (rejected, reason) if the transcript looks like a hallucination."""
    if no_speech_prob >= no_speech_threshold:
        return True, f"no_speech_prob={no_speech_prob:.2f}"

    if avg_logprob < logprob_threshold:
        return True, f"avg_logprob={avg_logprob:.2f}"

    if text.strip().lower().rstrip(".!?,") in _HALLUCINATION_PHRASES:
        return True, f"hallucination_blocklist"

    return False, ""


class State(enum.Enum):
    PASSIVE = "PASSIVE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    FOLLOW_UP = "FOLLOW_UP"


class StateMachine:
    """Five-state machine driving the full assistant pipeline."""

    def __init__(
        self,
        config: dict,
        capture: AudioCapture,
        player: AudioPlayer,
        vad: VoiceActivityDetector,
        utterance_detector: UtteranceDetector,
        wake_detector: WakeWordDetector,
        stt: WhisperSTT,
        llm_client: OpenRouterClient,
        tts: TTSEngine,
        session: Session,
        metrics: MetricsLogger,
    ):
        self._config = config
        self._capture = capture
        self._player = player
        self._vad = vad
        self._utterance_detector = utterance_detector
        self._wake_detector = wake_detector
        self._stt = stt
        self._llm = llm_client
        self._tts = tts
        self._session = session
        self._metrics = metrics
        metrics_cfg = config.get("metrics", {})
        self._log_transcripts = metrics_cfg.get("log_transcripts", False)
        self._log_llm_text = metrics_cfg.get("log_llm_text", False)

        self._state = State.PASSIVE
        self._running = False
        self._follow_up_deadline = 0.0

        # Barge-in tracking
        self._barge_in_enabled = config["vad"].get("barge_in_enabled", False)
        self._barge_in_count = 0
        self._barge_in_threshold = config["vad"].get("barge_in_frames", 8)
        self._speaking_start_time = 0.0
        self._barge_in_grace_s = config["vad"].get("barge_in_grace_s", 1.0)

        # Earcon settings
        self._earcon_sr = config["audio"]["sample_rate"]
        self._earcon_vol = config["earcon"].get("volume", 0.3)
        self._follow_up_grace_s = config["vad"].get("follow_up_grace_s", 0.3)
        self._follow_up_onset_frames = config["vad"].get("speech_onset_frames", 3)
        self._follow_up_start_time = 0.0
        self._listening_timeout_s = config["vad"].get("listening_timeout_s", 8.0)
        self._max_utterance_s = config["vad"].get("max_utterance_s", 30.0)
        self._listening_start_time = 0.0
        self._listening_hard_start = 0.0  # Never reset — absolute cap

        # Frame buffer for capturing speech onset before transition to LISTENING
        # 25 frames × 80ms = 2s of audio context
        self._recent_frames: list[tuple[np.ndarray, bool]] = []
        self._recent_frames_max = 25

        # Periodic reporting for dropped audio capture frames.
        self._capture_drop_report_s = config["audio"].get("capture_drop_report_s", 5.0)
        self._last_capture_drop_report_s = time.monotonic()

    @property
    def state(self) -> State:
        return self._state

    def _transition(self, new_state: State) -> None:
        old = self._state
        self._state = new_state
        print(f"  {_DIM}[{old.value}] → [{new_state.value}]{_RST}")
        self._metrics.log("state_transition", old=old.value, new=new_state.value)

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        """Main loop — start capture and process frames."""
        self._running = True
        self._capture.start()
        print(f"  {_DIM}State: [{self._state.value}] — say the wake word...{_RST}")

        while self._running:
            now = time.monotonic()
            if now - self._last_capture_drop_report_s >= self._capture_drop_report_s:
                self._report_capture_drops(now)
            frame = self._capture.get_frame(timeout=0.2)
            if frame is None:
                # No frame available — still check follow-up timeout
                if self._state == State.FOLLOW_UP:
                    self._check_follow_up_timeout()
                continue

            if self._state == State.PASSIVE:
                self._handle_passive(frame)
            elif self._state == State.LISTENING:
                self._handle_listening(frame)
            elif self._state == State.THINKING:
                pass  # Processing happens synchronously after LISTENING
            elif self._state == State.SPEAKING:
                self._handle_speaking(frame)
            elif self._state == State.FOLLOW_UP:
                self._handle_follow_up(frame)

    def _report_capture_drops(self, now_s: float) -> None:
        """Periodically emit capture drop counters for visibility."""
        self._last_capture_drop_report_s = now_s
        if not hasattr(self._capture, "consume_dropped_frames"):
            return
        dropped = self._capture.consume_dropped_frames()
        if dropped <= 0:
            return
        print(f"  {_YELLOW}Audio capture dropped {dropped} frame(s){_RST}")
        self._metrics.log("audio_frame_drop", dropped_frames=dropped)

    # ── State Handlers ──────────────────────────────────────────────

    def _handle_passive(self, frame) -> None:
        detected, score = self._wake_detector.process(frame)
        if detected:
            print(f"  {_YELLOW}Wake word detected {_DIM}(score={score:.2f}){_RST}")
            self._metrics.log("wake_detected", score=score)
            self._wake_detector.reset()

            # Play earcon and start LLM warmup
            play_earcon(self._player, self._config["earcon"], self._config["audio"]["sample_rate"])
            self._player.wait_until_done(timeout=0.5)
            self._llm.warmup()

            # Prepare for listening
            self._utterance_detector.reset()
            now = time.monotonic()
            self._listening_start_time = now
            self._listening_hard_start = now
            self._transition(State.LISTENING)

    def _handle_listening(self, frame) -> None:
        now = time.monotonic()

        # Hard cap — force-complete if listening has gone on too long (e.g. noisy room)
        if now - self._listening_hard_start >= self._max_utterance_s:
            if self._utterance_detector.state == "collecting":
                print(f"  {_YELLOW}Max utterance time reached, processing collected audio{_RST}")
                audio = self._utterance_detector.get_audio()
                play_named_earcon(self._player, "heard", self._earcon_sr, self._earcon_vol)
                self._player.wait_until_done(timeout=0.3)
                self._transition(State.THINKING)
                self._process_utterance(audio)
            else:
                print(f"  {_RED}Listening timed out, no speech detected{_RST}")
                self._metrics.log("listening_timeout")
                play_named_earcon(self._player, "goodbye", self._earcon_sr, self._earcon_vol)
                self._player.wait_until_done(timeout=0.5)
                self._session.clear()
                self._transition(State.PASSIVE)
                print(f"  {_DIM}State: [{self._state.value}] — say the wake word...{_RST}")
            return

        # Soft timeout — return to PASSIVE if no speech starts
        if now - self._listening_start_time >= self._listening_timeout_s:
            print(f"  {_RED}Listening timed out, no speech detected{_RST}")
            self._metrics.log("listening_timeout")
            play_named_earcon(self._player, "goodbye", self._earcon_sr, self._earcon_vol)
            self._player.wait_until_done(timeout=0.5)
            self._session.clear()
            self._transition(State.PASSIVE)
            print(f"  {_DIM}State: [{self._state.value}] — say the wake word...{_RST}")
            return

        is_speech = self._vad.is_speech(frame)
        state = self._utterance_detector.process(frame, is_speech)

        # Reset soft timeout once speech is confirmed (not single noisy frames)
        if self._utterance_detector.state == "collecting":
            self._listening_start_time = now

        if state == "complete":
            audio = self._utterance_detector.get_audio()
            play_named_earcon(self._player, "heard", self._earcon_sr, self._earcon_vol)
            self._player.wait_until_done(timeout=0.3)
            self._transition(State.THINKING)
            self._process_utterance(audio)

    def _process_utterance(self, audio) -> None:
        """Run STT → LLM → TTS pipeline (synchronous, runs in THINKING state)."""
        interaction_start = time.monotonic()
        detected_lang: str | None = None

        try:
            # STT
            stt_result = self._stt.transcribe(audio, self._config["audio"]["sample_rate"])
            transcript = stt_result["text"]
            detected_lang = stt_result.get("language")
            avg_logprob = stt_result["avg_logprob"]
            no_speech_prob = stt_result["no_speech_prob"]
            transcript_display = transcript if self._log_transcripts else f"<redacted:{len(transcript)} chars>"
            print(
                f"  {_CYAN}STT: \"{transcript_display}\"{_RST} "
                f"{_DIM}(lang={detected_lang}, {stt_result['transcription_time_s']:.2f}s, "
                f"logprob={avg_logprob:.2f}, no_speech={no_speech_prob:.2f}){_RST}"
            )
            self._metrics.log(
                "stt_complete",
                **stt_metrics_payload(stt_result, include_text=self._log_transcripts),
            )

            if not transcript.strip():
                print(f"  {_RED}Empty transcript, returning to follow-up{_RST}")
                self._enter_follow_up()
                return

            # Filter hallucinations
            stt_conf = self._config["stt"]
            rejected, reason = _check_hallucination(
                transcript, no_speech_prob, avg_logprob,
                no_speech_threshold=stt_conf.get("no_speech_threshold", 0.6),
                logprob_threshold=stt_conf.get("logprob_threshold", -1.0),
            )
            if rejected:
                print(f"  {_RED}STT rejected ({reason}), returning to follow-up{_RST}")
                rejected_payload = {"reason": reason, "text_chars": len(transcript)}
                if self._log_transcripts:
                    rejected_payload["text"] = transcript
                self._metrics.log("stt_rejected", **rejected_payload)
                self._enter_follow_up()
                return

            # LLM — use language-aware system prompt
            self._session.add_user_message(transcript)
            system_prompt = get_system_prompt(detected_lang)
            messages = build_messages(
                system_prompt,
                self._session.get_messages()[:-1],  # History before current message
                transcript,
            )
            llm_result = self._llm.chat(messages)
            raw_response_text = llm_result["text"]
            response_text = clean_for_tts(raw_response_text)
            llm_result = {**llm_result, "text": response_text}
            if response_text != raw_response_text:
                self._metrics.log(
                    "llm_response_sanitized",
                    raw_chars=len(raw_response_text),
                    clean_chars=len(response_text),
                    removed_chars=max(0, len(raw_response_text) - len(response_text)),
                )
            response_display = response_text if self._log_llm_text else f"<redacted:{len(response_text)} chars>"
            print(f"  {_BOLD}LLM: \"{response_display}\"{_RST}")
            print(f"       {_DIM}(ttft={llm_result['ttft_s']:.2f}s, total={llm_result['elapsed_s']:.2f}s){_RST}")
            self._metrics.log(
                "llm_complete",
                **llm_metrics_payload(llm_result, include_text=self._log_llm_text),
            )

            if not response_text.strip():
                print(f"  {_RED}Empty LLM response{_RST}")
                self._enter_follow_up()
                return

            self._session.add_assistant_message(response_text)

            # Use cleaned response for TTS.
            tts_text = response_text

            # Detect response language for TTS voice (may differ from input
            # when the user requests a translation)
            response_lang = detect_response_language(tts_text, fallback=detected_lang or "en")

            # TTS — use response language to select voice
            tts_start = time.monotonic()
            tts_audio, tts_sr = self._tts.synthesize(tts_text, language=response_lang)
            tts_elapsed = time.monotonic() - tts_start
            print(f"  {_DIM}TTS: synthesized in {tts_elapsed:.2f}s (input={detected_lang}, voice={response_lang}){_RST}")
            self._metrics.log("tts_complete", duration_s=tts_elapsed, input_language=detected_lang, voice_language=response_lang)

            # Log total pipeline latency
            total_elapsed = time.monotonic() - interaction_start
            self._metrics.log(
                "interaction_complete",
                total_elapsed_s=total_elapsed,
                stt_time_s=stt_result["transcription_time_s"],
                llm_ttft_s=llm_result["ttft_s"],
                llm_total_s=llm_result["elapsed_s"],
                tts_time_s=tts_elapsed,
                input_language=detected_lang,
                voice_language=response_lang,
            )

            # Play response
            self._barge_in_count = 0
            self._speaking_start_time = time.monotonic()
            self._player.play(tts_audio, sample_rate=tts_sr)
            self._transition(State.SPEAKING)

        except Exception as e:
            print(f"  {_RED}Pipeline error: {e}{_RST}")
            traceback.print_exc()
            self._metrics.log("pipeline_error", error=str(e))
            # Play error earcon, then try to speak the error
            play_named_earcon(self._player, "error", self._earcon_sr, self._earcon_vol)
            self._player.wait_until_done(timeout=0.5)
            try:
                err_msg = _ERROR_MESSAGES.get(detected_lang or "en", _ERROR_MESSAGES["en"])
                err_audio, err_sr = self._tts.synthesize(err_msg, language=detected_lang)
                self._player.play(err_audio, sample_rate=err_sr)
                self._player.wait_until_done(timeout=5)
            except Exception:
                pass
            self._enter_follow_up()

    def _handle_speaking(self, frame) -> None:
        # Check if playback finished
        if not self._player.is_playing:
            self._enter_follow_up()
            return

        if not self._barge_in_enabled:
            return

        # Grace period — ignore mic input right after playback starts
        # to avoid TTS audio from speakers triggering false barge-in
        if time.monotonic() - self._speaking_start_time < self._barge_in_grace_s:
            return

        # Buffer recent frames so speech onset isn't lost on barge-in
        is_speech = self._vad.is_speech(frame)
        self._recent_frames.append((frame.copy(), is_speech))
        if len(self._recent_frames) > self._recent_frames_max:
            self._recent_frames.pop(0)

        if is_speech:
            self._barge_in_count += 1
            if self._barge_in_count >= self._barge_in_threshold:
                print(f"  {_YELLOW}Barge-in detected!{_RST}")
                self._metrics.log("barge_in")
                self._player.stop()
                self._utterance_detector.reset()
                for buf_frame, buf_speech in self._recent_frames:
                    self._utterance_detector.process(buf_frame, buf_speech)
                self._recent_frames.clear()
                self._barge_in_count = 0
                now = time.monotonic()
                self._listening_start_time = now
                self._listening_hard_start = now
                self._transition(State.LISTENING)
        else:
            self._barge_in_count = 0

    def _handle_follow_up(self, frame) -> None:
        self._check_follow_up_timeout()
        if self._state != State.FOLLOW_UP:
            return

        # Always buffer frames so speech during grace period isn't lost
        is_speech = self._vad.is_speech(frame)
        self._recent_frames.append((frame.copy(), is_speech))
        if len(self._recent_frames) > self._recent_frames_max:
            self._recent_frames.pop(0)

        # Grace period — buffer frames but don't detect onset yet
        if time.monotonic() - self._follow_up_start_time < self._follow_up_grace_s:
            return

        if is_speech:
            self._barge_in_count += 1
            if self._barge_in_count >= self._follow_up_onset_frames:
                print(f"  {_YELLOW}Follow-up speech detected{_RST}")
                self._utterance_detector.reset()
                # Replay buffered frames so the start of speech is captured
                for buf_frame, buf_speech in self._recent_frames:
                    self._utterance_detector.process(buf_frame, buf_speech)
                self._recent_frames.clear()
                self._barge_in_count = 0
                now = time.monotonic()
                self._listening_start_time = now
                self._listening_hard_start = now
                self._transition(State.LISTENING)
        else:
            self._barge_in_count = 0

    def _check_follow_up_timeout(self) -> None:
        if time.monotonic() >= self._follow_up_deadline:
            self._session.clear()
            play_named_earcon(self._player, "goodbye", self._earcon_sr, self._earcon_vol)
            self._player.wait_until_done(timeout=0.5)
            self._transition(State.PASSIVE)
            print(f"  {_DIM}State: [{self._state.value}] — say the wake word...{_RST}")

    def _enter_follow_up(self) -> None:
        window = self._config["conversation"]["follow_up_window_s"]
        self._follow_up_deadline = time.monotonic() + window
        self._barge_in_count = 0
        self._recent_frames.clear()
        play_named_earcon(self._player, "ready", self._earcon_sr, self._earcon_vol)
        self._player.wait_until_done(timeout=0.5)
        self._follow_up_start_time = time.monotonic()
        self._transition(State.FOLLOW_UP)
