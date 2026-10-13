from copy import deepcopy
import json
from pathlib import Path
from string import Template
import os

from dcl_simulation import AudioSpec, ComponentInfo, ModelOutput


def create_model(config, *, config_dir):
    # config excludes "adapter"; config_dir is the configuration directory.
    return Model(config, config_dir=config_dir)


def load_model(config):
    # Load the SDK model using model_path, device, and generation from config.
    # Example: MyDuplex.from_pretrained(config["model_path"]).to(config["device"]).eval()
    raise NotImplementedError("Implement model loading in load_model")


def infer_tick(backend, state, input_audio, audio_spec, generation):
    # input_audio: mono float32 NumPy array of audio_spec.input_chunk_samples.
    # Keep KV caches, codec state, and audio buffers in state.
    # Return one output tick of matching duration as a CPU float32 NumPy array.
    # Process silent ticks too; emit np.zeros(audio_spec.output_chunk_samples, np.float32) if silent.
    # Buffer surplus SDK output for later ticks.
    # native_text contains newly generated text, or None if unavailable.
    # Reuse generation_unit_id throughout the same generation unit.
    # Example: return ModelOutput(audio=output_audio, native_text=new_text, generation_unit_id=unit_id)
    raise NotImplementedError("Implement one inference tick in infer_tick")


def new_state(backend):
    # Create any SDK session here and release it in close_state.
    return {}


def close_state(state):
    state.clear()


class Model:
    def __init__(self, config, *, config_dir=None):
        self.config = deepcopy(config)
        variables = dict(os.environ)
        variables.setdefault("MODEL_ROOT", str(Path.home() / "model"))
        path = Path(Template(config["model_path"]).substitute(variables)).expanduser()
        if not path.is_absolute() and config_dir is not None:
            path = config_dir / path
        self.config["model_path"] = str(path)
        self.audio_spec = AudioSpec(**config["audio"])
        self.provenance = ComponentInfo(config["name"], config["revision"], {
            "device": config["device"], "audio": deepcopy(config["audio"]),
            "generation": deepcopy(config["generation"]),
        })
        self._backend = load_model(self.config)

    @classmethod
    def from_file(cls, path):
        path = Path(path).expanduser().resolve()
        return cls(json.loads(path.read_text(encoding="utf-8")), config_dir=path.parent)

    def open_session(self):
        if self._backend is None:
            raise RuntimeError("model is closed")
        return ModelSession(self._backend, self.audio_spec, self.config["generation"])

    def close(self):
        # Call the SDK shutdown method, if any, before releasing the model.
        self._backend = None


class ModelSession:
    def __init__(self, backend, audio_spec, generation):
        self._backend = backend
        self.audio_spec = audio_spec
        self._generation = deepcopy(generation)
        self._state = new_state(backend)
        self._closed = False

    def process_tick(self, input_audio):
        if self._closed:
            raise RuntimeError("session is closed")
        return infer_tick(self._backend, self._state, input_audio, self.audio_spec, self._generation)

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                close_state(self._state)
            finally:
                self._state = None
                self._backend = None
