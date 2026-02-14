"""Central state machine orchestrating the voice assistant pipeline."""

import enum
import time
import traceback

from audio.capture import AudioCapture
from audio.playback import AudioPlayer
from audio.vad import VoiceActivityDetector, UtteranceDetector
from audio.earcon import play_earcon
from wake.detector import WakeWordDetector
from stt.whisper_stt import WhisperSTT
from llm.openrouter_client import OpenRouterClient
from llm.prompt import DEFAULT_SYSTEM_PROMPT, build_messages, clean_for_tts
from tts import TTSEngine
from assistant.session import Session
from assistant.metrics import MetricsLogger


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

        self._state = State.PASSIVE
        self._running = False
        self._follow_up_deadline = 0.0

        # Barge-in tracking
        self._barge_in_enabled = config["vad"].get("barge_in_enabled", False)
        self._barge_in_count = 0
        self._barge_in_threshold = config["vad"].get("barge_in_frames", 8)
        self._speaking_start_time = 0.0
        self._barge_in_grace_s = config["vad"].get("barge_in_grace_s", 1.0)

    @property
    def state(self) -> State:
        return self._state

    def _transition(self, new_state: State) -> None:
        old = self._state
        self._state = new_state
        print(f"  [{old.value}] → [{new_state.value}]")
        self._metrics.log("state_transition", old=old.value, new=new_state.value)

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        """Main loop — start capture and process frames."""
        self._running = True
        self._capture.start()
        print(f"  State: [{self._state.value}] — say the wake word...")

        while self._running:
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

    # ── State Handlers ──────────────────────────────────────────────

    def _handle_passive(self, frame) -> None:
        detected, score = self._wake_detector.process(frame)
        if detected:
            print(f"  Wake word detected (score={score:.2f})")
            self._metrics.log("wake_detected", score=score)
            self._wake_detector.reset()

            # Play earcon and start LLM warmup
            play_earcon(self._player, self._config["earcon"], self._config["audio"]["sample_rate"])
            self._player.wait_until_done(timeout=0.5)
            self._llm.warmup()

            # Prepare for listening
            self._utterance_detector.reset()
            self._transition(State.LISTENING)

    def _handle_listening(self, frame) -> None:
        is_speech = self._vad.is_speech(frame)
        state = self._utterance_detector.process(frame, is_speech)

        if state == "complete":
            audio = self._utterance_detector.get_audio()
            self._transition(State.THINKING)
            self._process_utterance(audio)

    def _process_utterance(self, audio) -> None:
        """Run STT → LLM → TTS pipeline (synchronous, runs in THINKING state)."""
        interaction_start = time.monotonic()

        try:
            # STT
            stt_result = self._stt.transcribe(audio, self._config["audio"]["sample_rate"])
            transcript = stt_result["text"]
            print(f"  STT: \"{transcript}\" ({stt_result['transcription_time_s']:.2f}s)")
            self._metrics.log("stt_complete", **stt_result)

            if not transcript.strip():
                print("  Empty transcript, returning to follow-up")
                self._enter_follow_up()
                return

            # LLM
            self._session.add_user_message(transcript)
            messages = build_messages(
                DEFAULT_SYSTEM_PROMPT,
                self._session.get_messages()[:-1],  # History before current message
                transcript,
            )
            llm_result = self._llm.chat(messages)
            response_text = llm_result["text"]
            print(f"  LLM: \"{response_text[:80]}{'...' if len(response_text) > 80 else ''}\"")
            print(f"       (ttft={llm_result['ttft_s']:.2f}s, total={llm_result['elapsed_s']:.2f}s)")
            self._metrics.log("llm_complete", **llm_result)

            if not response_text.strip():
                print("  Empty LLM response")
                self._enter_follow_up()
                return

            self._session.add_assistant_message(response_text)

            # Clean response for TTS (strip citations, URLs, markdown)
            tts_text = clean_for_tts(response_text)

            # TTS
            tts_start = time.monotonic()
            tts_audio, tts_sr = self._tts.synthesize(tts_text)
            tts_elapsed = time.monotonic() - tts_start
            print(f"  TTS: synthesized in {tts_elapsed:.2f}s")
            self._metrics.log("tts_complete", duration_s=tts_elapsed)

            # Log total pipeline latency
            total_elapsed = time.monotonic() - interaction_start
            self._metrics.log(
                "interaction_complete",
                total_elapsed_s=total_elapsed,
                stt_time_s=stt_result["transcription_time_s"],
                llm_ttft_s=llm_result["ttft_s"],
                llm_total_s=llm_result["elapsed_s"],
                tts_time_s=tts_elapsed,
            )

            # Play response
            self._barge_in_count = 0
            self._speaking_start_time = time.monotonic()
            self._player.play(tts_audio, sample_rate=tts_sr)
            self._transition(State.SPEAKING)

        except Exception as e:
            print(f"  Pipeline error: {e}")
            traceback.print_exc()
            self._metrics.log("pipeline_error", error=str(e))
            # Try to speak the error
            try:
                err_audio, err_sr = self._tts.synthesize("Sorry, something went wrong.")
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

        # Barge-in detection: monitor mic during playback
        is_speech = self._vad.is_speech(frame)
        if is_speech:
            self._barge_in_count += 1
            if self._barge_in_count >= self._barge_in_threshold:
                print("  Barge-in detected!")
                self._metrics.log("barge_in")
                self._player.stop()
                self._utterance_detector.reset()
                self._barge_in_count = 0
                self._transition(State.LISTENING)
        else:
            self._barge_in_count = 0

    def _handle_follow_up(self, frame) -> None:
        self._check_follow_up_timeout()
        if self._state != State.FOLLOW_UP:
            return

        # Listen for speech onset to continue conversation
        is_speech = self._vad.is_speech(frame)
        if is_speech:
            self._barge_in_count += 1
            if self._barge_in_count >= self._barge_in_threshold:
                print("  Follow-up speech detected")
                self._utterance_detector.reset()
                # Pre-seed the utterance detector with speech frames
                self._utterance_detector.process(frame, True)
                self._barge_in_count = 0
                self._transition(State.LISTENING)
        else:
            self._barge_in_count = 0

    def _check_follow_up_timeout(self) -> None:
        if time.monotonic() >= self._follow_up_deadline:
            self._session.clear()
            self._transition(State.PASSIVE)
            print(f"  State: [{self._state.value}] — say the wake word...")

    def _enter_follow_up(self) -> None:
        window = self._config["conversation"]["follow_up_window_s"]
        self._follow_up_deadline = time.monotonic() + window
        self._barge_in_count = 0
        self._transition(State.FOLLOW_UP)
