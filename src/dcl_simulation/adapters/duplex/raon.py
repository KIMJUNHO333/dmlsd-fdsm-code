from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math

import numpy as np

from ...audio import AudioSpec
from ...model import ModelMeasurements, ModelOutput
from ...records import ComponentInfo
from .._common import ConfigFile, model_path, positive, versions


def create_model(config, *, config_dir):
    return RaonModel(RaonConfig.from_dict(config, base=config_dir))


@dataclass(frozen=True)
class RaonConfig(ConfigFile):
    path_fields = ("model_path", "speaker_audio")
    model_path: str
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    system_prompt: str = "You are engaging in real-time conversation. Have a natural spoken conversation with the user in English."
    speaker_audio: str | None = None
    speak_first: bool = False
    do_sample: bool = True
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 66
    sil_penalty: float = 0.0
    bc_penalty: float = 0.0
    eos_penalty: float = 0.0
    max_sequence_length: int = 23612

    def __post_init__(self):
        model_path(self.model_path)
        ComponentInfo("KRAFTON/Raon-SpeechChat-9B", self.revision)
        if self.speaker_audio is not None:
            model_path(self.speaker_audio)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be nonempty")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("invalid dtype")
        if not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a string")
        for value in (self.speak_first, self.do_sample):
            if type(value) is not bool:
                raise ValueError("speak_first and do_sample must be booleans")
        positive(self.temperature, "temperature")
        if type(self.top_p) not in (int, float) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be between zero and one")
        for value in (self.top_k, self.max_sequence_length):
            if type(value) is not int or value <= 0:
                raise ValueError("top_k and max_sequence_length must be positive integers")
        for value in (self.sil_penalty, self.bc_penalty, self.eos_penalty):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("penalties must be finite")

    @property
    def generation(self):
        return {key: getattr(self, key) for key in (
            "speak_first", "do_sample", "temperature", "top_p", "top_k",
            "sil_penalty", "bc_penalty", "eos_penalty", "max_sequence_length",
        )}


class RaonModel:
    def __init__(self, config: RaonConfig):
        import torch
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        root = model_path(config.model_path)
        for path in (root / "config.json", root / "modeling_raon.py"):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.config = config
        self._session = None
        self._closed = False
        self._pipe = None
        self._speaker_embeds = None
        self._speaker_hash = None
        pipeline = get_class_from_dynamic_module("modeling_raon.RaonPipeline", str(root), local_files_only=True)
        try:
            self._pipe = pipeline(str(root), device=config.device, dtype=config.dtype, attn_implementation="sdpa")
            processor = self._pipe.processor
            rate = int(processor.sampling_rate)
            samples = Fraction(rate) / Fraction(str(processor.frame_rate))
            if samples.denominator != 1:
                raise ValueError("RAON frames must have an integer sample count")
            self.audio_spec = AudioSpec(rate, int(samples), rate, int(samples))
            if config.speaker_audio is not None:
                path = model_path(config.speaker_audio)
                self._speaker_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                with torch.inference_mode():
                    audio = self._pipe._load_speaker_audio(str(path))
                    self._speaker_embeds = self._pipe.model._compute_speaker_embeds(audio, None)
        except BaseException:
            self.close()
            raise

    @property
    def provenance(self):
        return ComponentInfo("KRAFTON/Raon-SpeechChat-9B", self.config.revision, {
            "device": self.config.device, "dtype": self.config.dtype, "attention": "sdpa",
            "system_prompt": self.config.system_prompt, "speaker_audio_sha256": self._speaker_hash,
            "generation": self.config.generation, "sample_rate": self.audio_spec.input_sample_rate,
            "frame_samples": self.audio_spec.input_chunk_samples,
            "packages": versions("transformers", "torch"),
        })

    def open_session(self):
        if self._closed:
            raise RuntimeError("RAON model is closed")
        if self._session is not None:
            raise RuntimeError("close the active RAON session before opening another")
        try:
            self._session = RaonSession(self)
        except BaseException:
            self.close()
            raise
        return self._session

    def close(self):
        self._closed = True
        try:
            if self._session is not None:
                self._session.close()
        finally:
            try:
                if self._pipe is not None:
                    self._pipe.model.stop_concurrent_audio_decoder()
            finally:
                self._pipe = self._speaker_embeds = None


class RaonSession:
    def __init__(self, model):
        import torch
        self._owner = model
        self.audio_spec = model.audio_spec
        self._closed = False
        pipe = model._pipe
        with torch.inference_mode():
            if model.config.system_prompt:
                inputs = pipe.processor([{"role": "system", "content": model.config.system_prompt}],
                    add_generation_prompt=False, device=model.config.device, dtype=pipe.dtype)
                tokens = inputs["input_ids"]
            else:
                tokens = torch.zeros((1, 0), dtype=torch.long, device=model.config.device)
            self._state = pipe.model.init_duplex_decoding_state(sequences=tokens,
                attention_mask=torch.ones_like(tokens), speaker_embeds=model._speaker_embeds,
                **model.config.generation)

    def process_tick(self, input_audio):
        import torch
        if self._closed:
            raise RuntimeError("RAON session is closed")
        self.audio_spec.validate_input(input_audio)
        pipe = self._owner._pipe
        before = int(self._state.sequences.shape[1])
        with torch.inference_mode():
            frame = torch.from_numpy(np.ascontiguousarray(input_audio)).to(self._owner.config.device)[None]
            self._state, waveform = pipe.model.duplex_decoding_step(state=self._state, audio_input=frame)
            if tuple(waveform.shape) != (1, self.audio_spec.output_chunk_samples):
                raise ValueError("RAON output does not match the tick size")
            audio = waveform[0].detach().float().cpu().numpy().copy()
            self.audio_spec.validate_output(audio)
            token_ids = tuple(int(token) for token in self._state.sequences[0, before:].tolist())
            special = set(pipe.processor.tokenizer.all_special_ids)
            text_ids = tuple(token for token in token_ids if 0 <= token < pipe.model.text_vocab_size and token not in special)
            text = pipe.processor.tokenizer.decode(list(text_ids), skip_special_tokens=True) if text_ids else ""
            return ModelOutput(audio, native_text=text, measurements=ModelMeasurements(
                native_tokens=token_ids, native_text_token_ids=text_ids,
                sequence_length=int(self._state.sequences.shape[1]), new_token_count=len(token_ids)))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._owner._pipe.model.free_duplex_decoding_state(self._state)
        finally:
            self._owner._session = None
            self._owner = self._state = None
