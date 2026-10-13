import json
import struct

import numpy as np

from .model import ModelOutput, ModelMeasurements


VERSION = 1
MAX_METADATA_BYTES = 1024 * 1024
MAX_AUDIO_BYTES = 64 * 1024 * 1024
HEADER = struct.Struct("!II")


class DuplexProtocolError(RuntimeError):
    pass


def _read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        chunk = stream.recv(size - len(result))
        if not chunk:
            raise EOFError("duplex connection closed")
        result.extend(chunk)
    return result


def send_message(stream, metadata, audio=None):
    data = json.dumps({**metadata, "version": VERSION}, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")
    payload = b"" if audio is None else audio.astype("<f4", copy=False).tobytes()
    if len(data) > MAX_METADATA_BYTES or len(payload) > MAX_AUDIO_BYTES:
        raise DuplexProtocolError("duplex message is too large")
    stream.sendall(HEADER.pack(len(data), len(payload)) + data + payload)


def receive_message(stream):
    metadata_size, audio_size = HEADER.unpack(_read_exact(stream, HEADER.size))
    if not 0 < metadata_size <= MAX_METADATA_BYTES or audio_size > MAX_AUDIO_BYTES or audio_size % 4:
        raise DuplexProtocolError("invalid duplex message size")
    try:
        metadata = json.loads(_read_exact(stream, metadata_size))
    except (ValueError, UnicodeError) as exc:
        raise DuplexProtocolError("invalid duplex metadata") from exc
    if not isinstance(metadata, dict) or type(metadata.get("version")) is not int or metadata["version"] != VERSION:
        raise DuplexProtocolError("unsupported duplex protocol version")
    audio = None
    if audio_size:
        audio = np.frombuffer(_read_exact(stream, audio_size), dtype="<f4").astype(np.float32, copy=True)
    return metadata, audio


def validate_output(output, spec):
    if not isinstance(output, ModelOutput):
        raise TypeError("process_tick must return a ModelOutput")
    spec.validate_output(output.audio)
    if output.native_text is not None and not isinstance(output.native_text, str):
        raise TypeError("native_text must be a string or None")
    if output.generation_unit_id is not None and (
            not isinstance(output.generation_unit_id, str) or not output.generation_unit_id.strip()):
        raise TypeError("generation_unit_id must be a nonempty string or None")
    if output.measurements is not None and not isinstance(output.measurements, ModelMeasurements):
        raise TypeError("measurements must be ModelMeasurements or None")
