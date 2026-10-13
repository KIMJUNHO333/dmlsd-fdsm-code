from importlib import import_module
import json
from pathlib import Path

from ...records import ComponentInfo


def load_duplex_model(path):
    source = Path(path).expanduser().resolve()
    values = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("duplex config must be a JSON object")
    target = values.pop("adapter", None)
    if not isinstance(target, str) or target.count(":") != 1:
        raise ValueError("adapter must be 'module:factory'")
    module, name = target.split(":")
    if not module or not name.isidentifier():
        raise ValueError("adapter must name an importable module and a factory function")
    factory = getattr(import_module(module), name)
    if not callable(factory):
        raise TypeError("adapter factory must be callable")
    model = factory(values, config_dir=source.parent)
    try:
        if not callable(getattr(model, "open_session", None)) or not callable(getattr(model, "close", None)):
            raise TypeError("duplex adapter must provide open_session() and close()")
        if not isinstance(model.provenance, ComponentInfo):
            raise TypeError("duplex adapter provenance must be ComponentInfo")
    except BaseException:
        close = getattr(model, "close", None)
        if callable(close):
            close()
        raise
    return model
