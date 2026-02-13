"""Wake word detection using openWakeWord."""

import numpy as np
from openwakeword.model import Model


class WakeWordDetector:
    """Wraps openWakeWord for streaming wake word detection.

    Expects 1280-sample int16 frames (80ms at 16kHz).
    """

    def __init__(self, wake_config: dict):
        self._model_name = wake_config["model_name"]
        self._threshold = wake_config["threshold"]
        self._model = Model(wakeword_models=[self._model_name], inference_framework="onnx")

    def process(self, frame_int16: np.ndarray) -> tuple[bool, float]:
        """Feed an audio frame and check for wake word detection.

        Returns (detected, score) where detected is True if the score
        exceeds the configured threshold.
        """
        prediction = self._model.predict(frame_int16)
        score = prediction.get(self._model_name, 0.0)
        detected = score >= self._threshold
        return detected, score

    def reset(self) -> None:
        """Clear internal buffers after a detection to avoid re-triggering."""
        self._model.reset()
