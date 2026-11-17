from copy import deepcopy
from dataclasses import dataclass

from ...records import ComponentInfo
from ...services import RecognizedWord, Transcription
from .._common import ConfigFile, audio_array, model_path, resample, sample_rate as check_rate, versions


@dataclass(frozen=True)
class ParakeetConfig(ConfigFile):
    model_path: str
    revision: str | None = None
    device: str = "cuda:0"

    def __post_init__(self):
        model_path(self.model_path)
        ComponentInfo("nvidia/parakeet-unified-en-0.6b", self.revision)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be nonempty")


class ParakeetASR:
    def __init__(self, config: ParakeetConfig):
        from nemo.collections.asr.models import ASRModel
        from omegaconf import open_dict
        path = model_path(config.model_path)
        if path.is_dir():
            path = path / "parakeet-unified-en-0.6b.nemo"
        if not path.is_file():
            raise FileNotFoundError(path)
        self.config = config
        self._model = ASRModel.restore_from(str(path), map_location=config.device).to(config.device).eval()
        self._model.preprocessor.featurizer.dither = 0.0
        self._model.preprocessor.featurizer.pad_to = 0
        self._model.encoder.set_default_att_context_size(att_context_size=[-1, -1, -1])
        self._decoding = deepcopy(self._model.cfg.decoding)
        with open_dict(self._decoding):
            self._decoding.strategy = "greedy_batch"
            self._decoding.greedy.loop_labels = True
            self._decoding.greedy.preserve_alignments = False
            self._decoding.fused_batch_size = -1
        self._model.change_decoding_strategy(self._decoding)

    @property
    def provenance(self):
        return ComponentInfo("nvidia/parakeet-unified-en-0.6b", self.config.revision,
            {"device": self.config.device, "language": "en", "sample_rate": 16000,
             "mode": "offline", "decoding": "greedy_batch", "word_timestamps": True,
             "packages": versions("nemo-toolkit", "torch", "scipy")})

    def transcribe(self, audio, *, sample_rate: int) -> Transcription:
        import torch
        if self._model is None:
            raise RuntimeError("ASR is closed")
        audio = audio_array(audio)
        check_rate(sample_rate)
        signal = resample(audio, sample_rate, 16000)
        try:
            with torch.inference_mode():
                results = self._model.transcribe([signal], batch_size=1, timestamps=True, verbose=False)
        finally:
            self._model.eval()
            self._model.change_decoding_strategy(self._decoding)
        if len(results) != 1:
            raise ValueError("ASR must return one hypothesis")
        hypothesis = results[0]
        if not isinstance(getattr(hypothesis, "text", None), str):
            raise ValueError("ASR hypothesis has no text")
        stamps = getattr(hypothesis, "timestamp", None)
        if not isinstance(stamps, dict) or "word" not in stamps:
            raise ValueError("ASR hypothesis has no word timestamps")
        words = []
        for item in stamps["word"] or ():
            text, start, end = item.get("word"), item.get("start"), item.get("end")
            if text is None or start is None or end is None:
                continue
            words.append(RecognizedWord(str(text), max(0, round(float(start) * sample_rate)),
                                       min(audio.size, max(0, round(float(end) * sample_rate)))))
        if hypothesis.text.strip() and not words:
            raise ValueError("ASR returned text without aligned words")
        return Transcription(hypothesis.text, tuple(words))

    def close(self):
        self._model = None
