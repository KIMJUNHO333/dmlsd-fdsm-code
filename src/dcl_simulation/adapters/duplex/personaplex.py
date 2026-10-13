from contextlib import ExitStack
from dataclasses import dataclass
import hashlib

from ...audio import AudioSpec
from ...records import ComponentInfo
from .._common import model_path, versions
from .moshi import MoshiConfig, MoshiModel, MoshiSession


def create_model(config, *, config_dir):
    return PersonaPlexModel(PersonaPlexConfig.from_dict(config, base=config_dir))


@dataclass(frozen=True)
class PersonaPlexConfig(MoshiConfig):
    path_fields = ("model_path", "voice_prompt_path")
    voice_prompt_path: str = ""
    text_prompt: str = "You enjoy having a good conversation."

    def __post_init__(self):
        super().__post_init__()
        model_path(self.voice_prompt_path)
        if not isinstance(self.text_prompt, str):
            raise ValueError("text_prompt must be a string")


class PersonaPlexModel(MoshiModel):
    def __init__(self, config: PersonaPlexConfig):
        import torch
        import sentencepiece
        from moshi.models import loaders, LMGen

        if not hasattr(LMGen, "step_system_prompts"):
            raise ImportError("PersonaPlex requires moshi-personaplex in a separate environment from moshi")
        root = model_path(config.model_path)
        voice = model_path(config.voice_prompt_path)
        for path in (root / "model.safetensors", root / "tokenizer-e351c8d8-checkpoint125.safetensors",
                     root / "tokenizer_spm_32k_3.model", voice):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.config = config
        self._voice = voice
        self._voice_hash = hashlib.sha256(voice.read_bytes()).hexdigest()
        self._session = None
        self._closed = False
        self._mimi = self._lm = self._tokenizer = None
        try:
            self._mimi = loaders.get_mimi(root / "tokenizer-e351c8d8-checkpoint125.safetensors", device=config.device)
            self._lm = loaders.get_moshi_lm(root / "model.safetensors", device=config.device,
                                          dtype=getattr(torch, config.dtype)).eval()
            self._tokenizer = sentencepiece.SentencePieceProcessor(str(root / "tokenizer_spm_32k_3.model"))
            self._output_codebooks = 8
            if self._mimi.sample_rate != 24000 or self._mimi.frame_rate != 12.5:
                raise ValueError("PersonaPlex requires 24 kHz, 12.5 fps Mimi")
            self.audio_spec = AudioSpec(24000, 1920, 24000, 1920)
        except BaseException:
            self.close()
            raise

    @property
    def provenance(self):
        return ComponentInfo("nvidia/personaplex-7b-v1", self.config.revision, {
            "device": self.config.device, "dtype": self.config.dtype,
            "generation": self.config.generation, "text_prompt": self.config.text_prompt,
            "voice_prompt": self._voice.name, "voice_prompt_sha256": self._voice_hash,
            "sample_rate": 24000, "frame_samples": 1920,
            "packages": versions("moshi-personaplex", "torch"),
        })

    def _new_session(self):
        return PersonaPlexSession(self)


class PersonaPlexSession(MoshiSession):
    def __init__(self, model):
        import torch
        from moshi.models import LMGen

        self.audio_spec = model.audio_spec
        self._model = model
        self._contexts = ExitStack()
        self._closed = False
        self._first_frame = False
        self._gen = None
        try:
            with torch.inference_mode():
                self._gen = LMGen(model._lm, audio_silence_frame_cnt=6, sample_rate=24000,
                    device=model.config.device, frame_rate=12.5, save_voice_prompt_embeddings=False,
                    **model.config.generation)
                self._contexts.enter_context(model._mimi.streaming(1))
                self._contexts.enter_context(self._gen.streaming(1))
                if model._voice.suffix == ".pt":
                    self._gen.load_voice_prompt_embeddings(str(model._voice))
                else:
                    self._gen.load_voice_prompt(str(model._voice))
                text = model.config.text_prompt.strip()
                if text and not (text.startswith("<system>") and text.endswith("<system>")):
                    text = f"<system> {text} <system>"
                self._gen.text_prompt_tokens = model._tokenizer.encode(text) if text else []
                self._gen.step_system_prompts(model._mimi)
                model._mimi.reset_streaming()
        except BaseException:
            self._contexts.close()
            self._gen = None
            self._model = None
            raise
