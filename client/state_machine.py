"""Client-side state machine for the voice assistant.

Runs on the Raspberry Pi. Handles wake word detection, audio capture,
VAD/utterance segmentation, earcon playback, and TTS chunk playback.
All heavy processing (STT, LLM, TTS synthesis) happens on the server.

States: PASSIVE -> LISTENING -> WAITING -> SPEAKING -> FOLLOW_UP
"""

import enum
import logging
import time
import traceback

import numpy as np

from shared import protocol
from client.audio.capture import AudioCapture
from client.audio.playback import AudioPlayer
from client.audio.vad import VoiceActivityDetector, UtteranceDetector
from client.audio.earcon import play_earcon, play_named_earcon
from client.wake.detector import WakeWordDetector
from client.connection import ServerConnection
from client.chunk_player import ChunkPlayer

log = logging.getLogger(__name__)

# ANSI color codes for terminal output
_DIM = "\033[90m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RST = "\033[0m"


class State(enum.Enum):
    PASSIVE = "PASSIVE"
    LISTENING = "LISTENING"
    WAITING = "WAITING"
    SPEAKING = "SPEAKING"
    FOLLOW_UP = "FOLLOW_UP"


class ClientStateMachine:
    """Five-state machine driving the client-side assistant pipeline."""

    def __init__(
        self,
        config: dict,
        capture: AudioCapture,
        player: AudioPlayer,
        vad: VoiceActivityDetector,
        utterance_detector: UtteranceDetector,
        wake_detector: WakeWordDetector,
        connection: ServerConnection,
        chunk_player: ChunkPlayer,
    ):
        self._config = config
        self._capture = capture
        self._player = player
        self._vad = vad
        self._utterance_detector = utterance_detector
        self._wake_detector = wake_detector
        self._conn = connection
        self._chunk_player = chunk_player

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
        self._listening_hard_start = 0.0

        # Frame buffer for capturing speech onset before transition
        self._recent_frames: list[tuple[np.ndarray, bool]] = []
        self._recent_frames_max = 25

        # Capture drop reporting
        self._capture_drop_report_s = config["audio"].get("capture_drop_report_s", 5.0)
        self._last_capture_drop_report_s = time.monotonic()

        # Audio device reconnection
        self._reconnect_delay_s = 1.0
        self._last_reconnect_attempt = 0.0

        # Follow-up window
        self._follow_up_window_s = config["conversation"].get("follow_up_window_s", 7.0)

    @property
    def state(self) -> State:
        return self._state

    def _transition(self, new_state: State) -> None:
        old = self._state
        self._state = new_state
        log.info("%s[%s] -> [%s]%s", _DIM, old.value, new_state.value, _RST)
        print(f"  {_DIM}[{old.value}] -> [{new_state.value}]{_RST}")

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        """Main loop — start capture and process frames + server messages."""
        self._running = True
        self._capture.start()
        print(f"  {_DIM}State: [{self._state.value}] -- say the wake word...{_RST}")

        while self._running:
            now = time.monotonic()

            # Report capture drops periodically
            if now - self._last_capture_drop_report_s >= self._capture_drop_report_s:
                self._report_capture_drops(now)

            # Check audio device health and reconnect if needed
            if not self._capture.is_healthy:
                self._try_reconnect_audio(now)

            # Check for incoming server messages (non-blocking)
            self._process_server_messages()

            # Get next audio frame
            frame = self._capture.get_frame(timeout=0.05)
            if frame is None:
                if self._state == State.FOLLOW_UP:
                    self._check_follow_up_timeout()
                continue

            # State-specific frame processing
            if self._state == State.PASSIVE:
                self._handle_passive(frame)
            elif self._state == State.LISTENING:
                self._handle_listening(frame)
            elif self._state == State.WAITING:
                pass  # Waiting for server response — handled via server messages
            elif self._state == State.SPEAKING:
                self._handle_speaking(frame)
            elif self._state == State.FOLLOW_UP:
                self._handle_follow_up(frame)

    def _report_capture_drops(self, now_s: float) -> None:
        self._last_capture_drop_report_s = now_s
        if not hasattr(self._capture, "consume_dropped_frames"):
            return
        dropped = self._capture.consume_dropped_frames()
        if dropped <= 0:
            return
        print(f"  {_YELLOW}Audio capture dropped {dropped} frame(s){_RST}")

    def _try_reconnect_audio(self, now: float) -> None:
        """Attempt to restart the audio stream after device loss."""
        if now - self._last_reconnect_attempt < self._reconnect_delay_s:
            return
        self._last_reconnect_attempt = now
        print(f"  {_YELLOW}Audio device lost — attempting reconnect...{_RST}")

        # Return to PASSIVE during reconnection
        if self._state != State.PASSIVE:
            self._transition(State.PASSIVE)

        if self._capture.restart():
            print(f"  {_CYAN}Audio device reconnected{_RST}")
            print(f"  {_DIM}State: [{self._state.value}] -- say the wake word...{_RST}")
        else:
            print(f"  {_RED}Reconnect failed — will retry in {self._reconnect_delay_s}s{_RST}")

    # ── Server message processing ───────────────────────────────

    def _process_server_messages(self) -> None:
        """Drain the server message queue and handle each message."""
        while True:
            try:
                msg = self._conn.recv_queue.get_nowait()
            except Exception:
                break

            if isinstance(msg, tuple):
                # (meta_dict, audio_ndarray) — TTS audio chunk
                meta, audio_int16 = msg
                self._on_tts_audio(meta, audio_int16)
            elif isinstance(msg, dict):
                self._dispatch_server_message(msg)

    def _dispatch_server_message(self, msg: dict) -> None:
        """Handle a parsed JSON message from the server."""
        msg_type = msg.get("type")

        if msg_type == protocol.WARMUP_ACK:
            log.debug("LLM warmup acknowledged")

        elif msg_type == protocol.STATUS:
            stage = msg.get("stage", "")
            log.info("Server status: %s", stage)
            if stage == protocol.STAGE_STT_COMPLETE:
                print(f"  {_DIM}STT complete{_RST}")
            elif stage == protocol.STAGE_LLM_COMPLETE:
                print(f"  {_DIM}LLM complete{_RST}")

        elif msg_type == protocol.STT_REJECTED:
            reason = msg.get("reason", "unknown")
            print(f"  {_RED}STT rejected ({reason}){_RST}")
            self._enter_follow_up()

        elif msg_type == protocol.TTS_DONE:
            cancelled = msg.get("cancelled", False)
            if not cancelled:
                self._chunk_player.finish_stream()
            log.info("TTS done (cancelled=%s)", cancelled)
            # If we're still WAITING and no audio was streamed, go to follow-up
            if self._state == State.WAITING:
                self._enter_follow_up()

        elif msg_type == protocol.SESSION_CLEARED:
            log.info("Session cleared by server")

        elif msg_type == protocol.ERROR:
            error_msg = msg.get("message", "unknown error")
            stage = msg.get("stage", "")
            print(f"  {_RED}Server error ({stage}): {error_msg}{_RST}")
            # If we're waiting, enter follow-up
            if self._state == State.WAITING:
                self._enter_follow_up()

    def _on_tts_audio(self, meta: dict, audio_int16: np.ndarray) -> None:
        """Handle a TTS audio chunk from the server."""
        sample_rate = meta.get("sample_rate", 22050)
        chunk_index = meta.get("chunk_index", 0)
        is_last = meta.get("is_last", False)

        log.debug("TTS chunk %d: %d samples at %d Hz (last=%s)",
                  chunk_index, len(audio_int16), sample_rate, is_last)

        if self._state == State.WAITING:
            # First chunk — transition to SPEAKING and start playback
            self._chunk_player.start_stream()
            self._barge_in_count = 0
            self._speaking_start_time = time.monotonic()
            self._transition(State.SPEAKING)

        self._chunk_player.enqueue(audio_int16, sample_rate)

        if is_last:
            self._chunk_player.finish_stream()

    # ── State Handlers ──────────────────────────────────────────

    def _handle_passive(self, frame) -> None:
        detected, score = self._wake_detector.process(frame)
        if detected:
            print(f"  {_YELLOW}Wake word detected {_DIM}(score={score:.2f}){_RST}")
            self._wake_detector.reset()

            # Play earcon
            play_earcon(self._player, self._config["earcon"], self._config["audio"]["sample_rate"])
            self._player.wait_until_done(timeout=0.5)

            # Notify server (triggers LLM warmup)
            self._conn.send_wake(score)

            # Prepare for listening
            self._utterance_detector.reset()
            now = time.monotonic()
            self._listening_start_time = now
            self._listening_hard_start = now
            self._transition(State.LISTENING)

    def _handle_listening(self, frame) -> None:
        now = time.monotonic()

        # Hard cap
        if now - self._listening_hard_start >= self._max_utterance_s:
            if self._utterance_detector.state == "collecting":
                print(f"  {_YELLOW}Max utterance time reached, sending collected audio{_RST}")
                audio = self._utterance_detector.get_audio()
                play_named_earcon(self._player, "heard", self._earcon_sr, self._earcon_vol)
                self._player.wait_until_done(timeout=0.3)
                self._send_utterance(audio)
            else:
                print(f"  {_RED}Listening timed out, no speech detected{_RST}")
                play_named_earcon(self._player, "goodbye", self._earcon_sr, self._earcon_vol)
                self._player.wait_until_done(timeout=0.5)
                self._conn.send_follow_up_timeout()
                self._transition(State.PASSIVE)
                print(f"  {_DIM}State: [{self._state.value}] -- say the wake word...{_RST}")
            return

        # Soft timeout
        if now - self._listening_start_time >= self._listening_timeout_s:
            print(f"  {_RED}Listening timed out, no speech detected{_RST}")
            play_named_earcon(self._player, "goodbye", self._earcon_sr, self._earcon_vol)
            self._player.wait_until_done(timeout=0.5)
            self._conn.send_follow_up_timeout()
            self._transition(State.PASSIVE)
            print(f"  {_DIM}State: [{self._state.value}] -- say the wake word...{_RST}")
            return

        is_speech = self._vad.is_speech(frame)
        state = self._utterance_detector.process(frame, is_speech)

        # Reset soft timeout once speech is confirmed
        if self._utterance_detector.state == "collecting":
            self._listening_start_time = now

        if state == "complete":
            audio = self._utterance_detector.get_audio()
            play_named_earcon(self._player, "heard", self._earcon_sr, self._earcon_vol)
            self._player.wait_until_done(timeout=0.3)
            self._send_utterance(audio)

    def _send_utterance(self, audio: np.ndarray) -> None:
        """Send utterance audio to server and transition to WAITING."""
        sample_rate = self._config["audio"]["sample_rate"]
        self._conn.send_utterance(audio, sample_rate)
        self._transition(State.WAITING)

    def _handle_speaking(self, frame) -> None:
        # Check if chunk player finished
        if not self._chunk_player.is_playing:
            self._enter_follow_up()
            return

        if not self._barge_in_enabled:
            return

        # Grace period
        if time.monotonic() - self._speaking_start_time < self._barge_in_grace_s:
            return

        # Buffer recent frames
        is_speech = self._vad.is_speech(frame)
        self._recent_frames.append((frame.copy(), is_speech))
        if len(self._recent_frames) > self._recent_frames_max:
            self._recent_frames.pop(0)

        if is_speech:
            self._barge_in_count += 1
            if self._barge_in_count >= self._barge_in_threshold:
                print(f"  {_YELLOW}Barge-in detected!{_RST}")
                self._chunk_player.cancel()
                self._conn.send_barge_in()
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

        # Buffer frames
        is_speech = self._vad.is_speech(frame)
        self._recent_frames.append((frame.copy(), is_speech))
        if len(self._recent_frames) > self._recent_frames_max:
            self._recent_frames.pop(0)

        # Grace period
        if time.monotonic() - self._follow_up_start_time < self._follow_up_grace_s:
            return

        if is_speech:
            self._barge_in_count += 1
            if self._barge_in_count >= self._follow_up_onset_frames:
                print(f"  {_YELLOW}Follow-up speech detected{_RST}")
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

    def _check_follow_up_timeout(self) -> None:
        if time.monotonic() >= self._follow_up_deadline:
            self._conn.send_follow_up_timeout()
            play_named_earcon(self._player, "goodbye", self._earcon_sr, self._earcon_vol)
            self._player.wait_until_done(timeout=0.5)
            self._transition(State.PASSIVE)
            print(f"  {_DIM}State: [{self._state.value}] -- say the wake word...{_RST}")

    def _enter_follow_up(self) -> None:
        self._follow_up_deadline = time.monotonic() + self._follow_up_window_s
        self._barge_in_count = 0
        self._recent_frames.clear()
        play_named_earcon(self._player, "ready", self._earcon_sr, self._earcon_vol)
        self._player.wait_until_done(timeout=0.5)
        self._follow_up_start_time = time.monotonic()
        self._transition(State.FOLLOW_UP)
