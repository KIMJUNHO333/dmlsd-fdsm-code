from dataclasses import dataclass

import numpy as np

from ...records import ComponentInfo
from .._common import ConfigFile, audio_array, model_path, versions


@dataclass(frozen=True)
class ECAPAConfig(ConfigFile):
    model_path: str
    revision: str | None = None
    device: str = "cpu"

    def __post_init__(self):
        model_path(self.model_path)
        ComponentInfo("speechbrain/spkrec-ecapa-voxceleb", self.revision)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be nonempty")


class ECAPAEmbedder:
    def __init__(self, config: ECAPAConfig):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        path = model_path(config.model_path)
        if not (path / "hyperparams.yaml").is_file():
            raise FileNotFoundError(path / "hyperparams.yaml")
        self.config = config
        self._model = EncoderClassifier.from_hparams(
            source=str(path), savedir=str(path), overrides={"pretrained_path": str(path)},
            run_opts={"device": config.device}, local_strategy=LocalStrategy.NO_LINK,
        )
        self._model.mods.eval()

    @property
    def provenance(self):
        return ComponentInfo("speechbrain/spkrec-ecapa-voxceleb", self.config.revision,
            {"device": self.config.device, "sample_rate": 16000,
             "packages": versions("speechbrain", "torch")})

    def embed(self, audio, *, sample_rate):
        import torch
        if self._model is None:
            raise RuntimeError("speaker encoder is closed")
        if sample_rate != 16000:
            raise ValueError("speaker encoder requires 16 kHz audio")
        audio = audio_array(audio)
        with torch.inference_mode():
            vector = self._model.encode_batch(torch.from_numpy(audio).unsqueeze(0))
        vector = np.asarray(vector.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError("invalid speaker embedding")
        return vector

    def close(self):
        self._model = None
