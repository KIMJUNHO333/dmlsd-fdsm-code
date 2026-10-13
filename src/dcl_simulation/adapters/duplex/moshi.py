from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ...audio import AudioSpec
from ...model import ModelMeasurements, ModelOutput
from ...records import ComponentInfo
from .._common import ConfigFile, model_path, positive, versions


def create_model(config, *, config_dir):
    return MoshiModel(MoshiConfig.from_dict(config, base=config_dir))


@dataclass(frozen=True)
class MoshiConfig(ConfigFile):
    model_path: str
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    use_sampling: bool = True
    audio_temperature: float = 0.8
    text_temperature: float = 0.7
    audio_top_k: int = 250
    text_top_k: int = 25

    def __post_init__(self):
        model_path(self.model_path)
        ComponentInfo("kyutai/moshiko-pytorch-bf16", self.revision)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be nonempty")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError("dtype must be bfloat16, float16 or float32")
        if type(self.use_sampling) is not bool:
            raise ValueError("use_sampling must be a boolean")
        positive(self.audio_temperature, "audio_temperature")
        positive(self.text_temperature, "text_temperature")
        for value in (self.audio_top_k, self.text_top_k):
            if type(value) is not int or value <= 0:
                raise ValueError("top_k must be a positive integer")

    @property
    def generation(self):
        return dict(use_sampling=self.use_sampling, temp=self.audio_temperature,
                    temp_text=self.text_temperature, top_k=self.audio_top_k,
                    top_k_text=self.text_top_k)


class MoshiModel:
    def __init__(self, config: MoshiConfig):
        import torch
        from moshi.models.loaders import CheckpointInfo

        root = model_path(config.model_path)
        paths = dict(moshi_weights=root / "model.safetensors",
                     mimi_weights=root / "tokenizer-e351c8d8-checkpoint125.safetensors",
                     tokenizer=root / "tokenizer_spm_32k_3.model")
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        if (root / "config.json").exists():
            raise ValueError("this adapter requires the original Moshiko 7B checkpoint without config.json")
        self.config = config
        self._session = None
        self._closed = False
        self._mimi = self._lm = self._tokenizer = None
        checkpoint = CheckpointInfo(**paths)
        try:
            self._mimi = checkpoint.get_mimi(device=config.device)
            self._lm = checkpoint.get_moshi(device=config.device, dtype=getattr(torch, config.dtype))
            self._output_codebooks = self._lm.dep_q
            self._tokenizer = checkpoint.get_text_tokenizer()
            rate = int(self._mimi.sample_rate)
            samples = Fraction(rate) / Fraction(str(self._mimi.frame_rate))
            if self._mimi.channels != 1 or samples.denominator != 1:
                raise ValueError("Mimi must produce mono frames with an integer sample count")
            self.audio_spec = AudioSpec(rate, int(samples), rate, int(samples))
        except BaseException:
            self.close()
            raise

    @property
    def provenance(self):
        return ComponentInfo("kyutai/moshiko-pytorch-bf16", self.config.revision, {
            "device": self.config.device, "dtype": self.config.dtype,
            "generation": self.config.generation,
            "sample_rate": self.audio_spec.input_sample_rate,
            "frame_samples": self.audio_spec.input_chunk_samples,
            "packages": versions("moshi", "torch"),
        })

    def open_session(self):
        if self._closed:
            raise RuntimeError("Moshi model is closed")
        if self._session is not None:
            raise RuntimeError("close the active Moshi session before opening another")
        try:
            self._session = self._new_session()
        except BaseException:
            self.close()
            raise
        return self._session

    def _new_session(self):
        return MoshiSession(self)

    def close(self):
        self._closed = True
        try:
            if self._session is not None:
                self._session.close()
        finally:
            self._mimi = self._lm = self._tokenizer = None


class MoshiSession:
    def __init__(self, model):
        import torch
        from moshi.models import LMGen

        self.audio_spec = model.audio_spec
        self._model = model
        self._contexts = ExitStack()
        self._closed = False
        self._first_frame = True
        self._gen = None
        try:
            with torch.inference_mode():
                self._gen = LMGen(model._lm, **model.config.generation)
                self._contexts.enter_context(model._mimi.streaming(1))
                self._contexts.enter_context(self._gen.streaming(1))
        except BaseException:
            self._contexts.close()
            self._gen = None
            self._model = None
            raise

    def process_tick(self, input_audio):
        import torch
        if self._closed:
            raise RuntimeError("Moshi session is closed")
        self.audio_spec.validate_input(input_audio)
        model = self._model
        with torch.inference_mode():
            frame = torch.from_numpy(np.ascontiguousarray(input_audio)).to(model.config.device)[None, None]
            codes = model._mimi.encode(frame)
            if self._first_frame:
                self._gen.step(codes)
                self._first_frame = False
            tokens = self._gen.step(codes)
            if tokens is None:
                return ModelOutput(np.zeros(self.audio_spec.output_chunk_samples, dtype=np.float32),
                                   native_text="", measurements=ModelMeasurements(
                                       native_tokens=(), native_text_token_ids=(), new_token_count=0))
            if tuple(tokens.shape) != (1, model._lm.dep_q + 1, 1):
                raise ValueError("Moshi must return one frame of text and audio tokens")
            pcm = model._mimi.decode(tokens[:, 1:1 + model._output_codebooks])
            if tuple(pcm.shape) != (1, 1, self.audio_spec.output_chunk_samples):
                raise ValueError("Mimi output does not match the tick size")
            audio = pcm[0, 0].detach().float().cpu().numpy().copy()
            self.audio_spec.validate_output(audio)
            token_ids = tuple(int(value) for value in tokens[0, :, 0].detach().cpu().tolist())
            text_id = token_ids[0]
            text = "" if text_id in (0, 3) else model._tokenizer.id_to_piece(text_id).replace("▁", " ")
            return ModelOutput(audio, native_text=text, measurements=ModelMeasurements(
                native_tokens=token_ids, native_text_token_ids=(text_id,), new_token_count=len(token_ids)))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._contexts.close()
        finally:
            if self._model is not None:
                self._model._session = None
            self._gen = None
            self._model = None
