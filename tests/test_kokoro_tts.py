"""Tests for KokoroTTS engine with mocked kokoro.KPipeline."""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Stub kokoro + piper before importing KokoroTTS ───────────────

def _make_fake_pipeline_cls():
    """Return a FakeKPipeline class that yields deterministic audio."""

    class FakeKPipeline:
        """Mimics kokoro.KPipeline: __init__(lang_code=...), __call__(text, voice=..., speed=...)."""

        def __init__(self, lang_code="a"):
            self.lang_code = lang_code
            self.calls: list[dict] = []

        def __call__(self, text, voice="af_bella", speed=1.0):
            self.calls.append({"text": text, "voice": voice, "speed": speed})
            # Yield one (graphemes, phonemes, audio) tuple per call
            audio = np.sin(np.linspace(0, 2 * np.pi, 480, endpoint=False)).astype(np.float32)
            yield text, "phonemes", audio

    return FakeKPipeline


def _install_stubs():
    """Install kokoro and piper stubs into sys.modules."""
    FakeKPipeline = _make_fake_pipeline_cls()

    kokoro_mod = types.ModuleType("kokoro")
    kokoro_mod.KPipeline = FakeKPipeline
    sys.modules["kokoro"] = kokoro_mod

    if "piper" not in sys.modules:
        sys.modules["piper"] = types.ModuleType("piper")
    if "piper.voice" not in sys.modules:
        pv = types.ModuleType("piper.voice")
        pv.PiperVoice = type("PiperVoice", (), {})
        sys.modules["piper.voice"] = pv
    if "piper.config" not in sys.modules:
        pc = types.ModuleType("piper.config")
        pc.SynthesisConfig = type("SynthesisConfig", (), {"__init__": lambda *a, **kw: None})
        sys.modules["piper.config"] = pc

    # Clear cached server.tts modules so they pick up the stubs
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("server.tts"):
            sys.modules.pop(mod_name, None)


_install_stubs()

from server.tts.kokoro_tts import KokoroTTS, _split_sentences, _SAMPLE_RATE, _UNSUPPORTED_LANGS


# ── Fixtures ─────────────────────────────────────────────────────

def _base_config(**overrides):
    cfg = {
        "engine": "kokoro",
        "default_language": "en",
        "sentence_silence": 0.2,
        "speed": 1.0,
        "model_dir": "models/piper",
        "voices": {
            "en": {"kokoro_voice": "af_bella"},
        },
    }
    cfg.update(overrides)
    return cfg


# ── Unit tests ───────────────────────────────────────────────────

class TestSplitSentences:
    def test_single_sentence(self):
        assert _split_sentences("Hello world") == ["Hello world"]

    def test_multiple_sentences(self):
        result = _split_sentences("First. Second! Third?")
        assert result == ["First.", "Second!", "Third?"]

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_whitespace_only(self):
        assert _split_sentences("   ") == []

    def test_no_punctuation(self):
        assert _split_sentences("no ending punctuation") == ["no ending punctuation"]


class TestKokoroTTSInit:
    def test_basic_init(self):
        tts = KokoroTTS(_base_config())
        assert "en" in tts._kokoro_voices
        assert tts._kokoro_voices["en"] == "af_bella"
        assert "a" in tts._pipelines  # 'a' = American English

    def test_no_voices_raises(self):
        with pytest.raises(RuntimeError, match="no voices configured"):
            KokoroTTS(_base_config(voices={}))

    def test_unknown_language_skipped(self):
        cfg = _base_config(voices={"xx": {"kokoro_voice": "xx_voice"}})
        with pytest.raises(RuntimeError, match="no voices configured"):
            KokoroTTS(cfg)

    def test_no_piper_fallback_when_no_piper_voice(self):
        tts = KokoroTTS(_base_config())
        assert tts._piper_fallback is None

    def test_speed_from_config(self):
        tts = KokoroTTS(_base_config(speed=1.5))
        assert tts._speed == 1.5


class TestSynthesizeChunks:
    def test_single_sentence_yields_one_chunk(self):
        tts = KokoroTTS(_base_config())
        chunks = list(tts.synthesize_chunks("Hello world."))
        assert len(chunks) == 1
        audio, sr, is_last = chunks[0]
        assert sr == _SAMPLE_RATE
        assert is_last is True
        assert audio.dtype == np.int16
        assert len(audio) > 0

    def test_multiple_sentences_yield_multiple_chunks(self):
        tts = KokoroTTS(_base_config())
        chunks = list(tts.synthesize_chunks("First sentence. Second sentence."))
        assert len(chunks) == 2
        assert chunks[0][2] is False  # is_last
        assert chunks[1][2] is True   # is_last

    def test_inter_sentence_silence_added(self):
        tts = KokoroTTS(_base_config(sentence_silence=0.1))
        chunks = list(tts.synthesize_chunks("A. B."))
        # First chunk should have silence appended (0.1s * 24000 = 2400 extra samples)
        audio_first = chunks[0][0]
        audio_last = chunks[1][0]
        # First chunk has silence, last doesn't — first should be longer
        assert len(audio_first) > len(audio_last)

    def test_empty_text_yields_nothing(self):
        tts = KokoroTTS(_base_config())
        chunks = list(tts.synthesize_chunks(""))
        assert chunks == []

    def test_clipping_applied(self):
        """Audio exceeding [-1, 1] should be clipped before int16 conversion."""
        tts = KokoroTTS(_base_config())

        # Monkey-patch the pipeline to return audio outside [-1, 1]
        loud_audio = np.array([1.5, -1.5, 0.5], dtype=np.float32)
        fake_pipeline = tts._pipelines["a"]
        original_call = fake_pipeline.__call__
        fake_pipeline.__call__ = lambda text, voice=None, speed=1.0: iter([("g", "p", loud_audio)])

        chunks = list(tts.synthesize_chunks("Loud."))
        audio_int16 = chunks[0][0]
        # Values should be clipped to [-32767, 32767]
        assert audio_int16.max() <= 32767
        assert audio_int16.min() >= -32767

    def test_sample_rate_is_24000(self):
        tts = KokoroTTS(_base_config())
        chunks = list(tts.synthesize_chunks("Test."))
        for _, sr, _ in chunks:
            assert sr == 24000

    def test_speed_passed_to_pipeline(self):
        tts = KokoroTTS(_base_config(speed=1.3))
        list(tts.synthesize_chunks("Hello."))
        pipeline = tts._pipelines["a"]
        assert pipeline.calls[0]["speed"] == 1.3

    def test_voice_name_passed_to_pipeline(self):
        tts = KokoroTTS(_base_config())
        list(tts.synthesize_chunks("Hello."))
        pipeline = tts._pipelines["a"]
        assert pipeline.calls[0]["voice"] == "af_bella"


class TestSynthesize:
    def test_returns_float32_and_sample_rate(self):
        tts = KokoroTTS(_base_config())
        audio, sr = tts.synthesize("Hello.")
        assert audio.dtype == np.float32
        assert sr == _SAMPLE_RATE

    def test_empty_text(self):
        tts = KokoroTTS(_base_config())
        audio, sr = tts.synthesize("")
        assert len(audio) == 0


class TestUnsupportedLanguageFallback:
    def test_de_without_piper_fallback_uses_kokoro_voice(self):
        """When no Piper fallback is configured, unsupported lang routes to Kokoro."""
        tts = KokoroTTS(_base_config())
        assert tts._piper_fallback is None

        chunks = list(tts.synthesize_chunks("Hallo Welt.", language="de"))
        assert len(chunks) >= 1
        _, sr, _ = chunks[0]
        assert sr == _SAMPLE_RATE  # Kokoro rate, not Piper

    def test_de_with_default_de_no_infinite_recursion(self):
        """Regression: default_language='de' + no Piper must not recurse forever."""
        cfg = _base_config(default_language="de")
        tts = KokoroTTS(cfg)
        # This would previously cause RecursionError
        chunks = list(tts.synthesize_chunks("Hallo.", language="de"))
        assert len(chunks) >= 1

    def test_de_delegates_to_piper_fallback(self):
        """When Piper fallback is configured, unsupported lang delegates to it."""
        tts = KokoroTTS(_base_config())

        # Install a mock Piper fallback
        mock_piper = MagicMock()
        mock_piper.synthesize_chunks.return_value = iter([
            (np.array([1, 2, 3], dtype=np.int16), 22050, True),
        ])
        tts._piper_fallback = mock_piper

        chunks = list(tts.synthesize_chunks("Hallo.", language="de"))
        assert len(chunks) == 1
        assert chunks[0][1] == 22050  # Piper sample rate
        mock_piper.synthesize_chunks.assert_called_once_with("Hallo.", language="de")

    def test_synthesize_returns_piper_sample_rate_for_de(self):
        """synthesize() must return Piper's sample rate when delegating to Piper."""
        tts = KokoroTTS(_base_config())

        mock_piper = MagicMock()
        mock_piper.synthesize_chunks.return_value = iter([
            (np.array([100, 200], dtype=np.int16), 22050, True),
        ])
        tts._piper_fallback = mock_piper

        audio, sr = tts.synthesize("Hallo.", language="de")
        assert sr == 22050  # NOT 24000


class TestResolve:
    def test_known_language(self):
        tts = KokoroTTS(_base_config())
        pipeline, voice = tts._resolve("en")
        assert voice == "af_bella"

    def test_unknown_language_falls_back_to_default(self):
        tts = KokoroTTS(_base_config())
        pipeline, voice = tts._resolve("xx")
        assert voice == "af_bella"  # falls back to default (en)

    def test_none_uses_default(self):
        tts = KokoroTTS(_base_config())
        pipeline, voice = tts._resolve(None)
        assert voice == "af_bella"


class TestTTSEngineProtocol:
    def test_kokoro_satisfies_protocol(self):
        from server.tts import TTSEngine
        tts = KokoroTTS(_base_config())
        assert isinstance(tts, TTSEngine)


class TestCreateTTSFactory:
    def test_factory_creates_kokoro(self):
        from server.tts import create_tts
        tts = create_tts(_base_config())
        assert isinstance(tts, KokoroTTS)

    def test_factory_unknown_engine_raises(self):
        from server.tts import create_tts
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            create_tts({"engine": "nonexistent"})
