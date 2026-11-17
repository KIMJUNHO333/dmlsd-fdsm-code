from dataclasses import dataclass
import json
import math
from pathlib import Path
import socket

from .audio import AudioSpec
from .model import ModelMeasurements, ModelOutput
from .records import ComponentInfo
from ._duplex_protocol import DuplexProtocolError, receive_message, send_message, validate_output


@dataclass(frozen=True)
class DuplexClientConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    timeout_s: float = 300.0

    def __post_init__(self):
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be nonempty")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if type(self.timeout_s) not in (int, float) or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(Path(path).expanduser().read_text(encoding="utf-8")))


class DuplexRemoteError(RuntimeError):
    def __init__(self, operation, error_type, message):
        self.operation = operation
        self.remote_type = error_type
        super().__init__(f"{operation}: {error_type}: {message}")


class DuplexClient:
    def __init__(self, config=DuplexClientConfig()):
        self.config = config
        self._session = None
        self._socket = socket.create_connection((config.host, config.port), timeout=config.timeout_s)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            response, _ = self._request("info")
            self.provenance = ComponentInfo(**response["provenance"])
        except BaseException:
            self._disconnect()
            raise

    def _disconnect(self):
        stream, self._socket = self._socket, None
        if self._session is not None:
            self._session._closed = True
            self._session = None
        if stream is not None:
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            stream.close()

    def _request(self, operation, audio=None):
        if self._socket is None:
            raise RuntimeError("duplex client is closed")
        try:
            send_message(self._socket, {"op": operation}, audio)
            response, output = receive_message(self._socket)
            if response.get("ok") is False:
                error = response["error"]
                raise DuplexRemoteError(operation, error["type"], error["message"])
            if response.get("ok") is not True or response.get("op") != operation:
                raise DuplexProtocolError("unexpected duplex response")
            if operation != "tick" and output is not None:
                raise DuplexProtocolError("unexpected response audio")
            return response, output
        except BaseException:
            self._disconnect()
            raise

    def open_session(self):
        if self._session is not None:
            raise RuntimeError("a duplex session is already open")
        try:
            response, _ = self._request("open")
            self._session = DuplexClientSession(self, AudioSpec(**response["audio_spec"]))
            return self._session
        except BaseException:
            self._disconnect()
            raise

    def close(self):
        try:
            if self._session is not None:
                self._session.close()
        finally:
            self._disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class DuplexClientSession:
    def __init__(self, client, audio_spec):
        self._client = client
        self.audio_spec = audio_spec
        self._closed = False

    def process_tick(self, input_audio):
        if self._closed:
            raise RuntimeError("duplex session is closed")
        self.audio_spec.validate_input(input_audio)
        try:
            response, audio = self._client._request("tick", input_audio)
            measurements = response["measurements"]
            if measurements is not None:
                measurements = dict(measurements)
                for field in ("native_tokens", "native_text_token_ids"):
                    if measurements.get(field) is not None:
                        measurements[field] = tuple(measurements[field])
                measurements = ModelMeasurements(**measurements)
            output = ModelOutput(audio, response["native_text"], response["generation_unit_id"], measurements)
            validate_output(output, self.audio_spec)
            return output
        except BaseException:
            self._client._disconnect()
            raise

    def close(self):
        if self._closed:
            return
        try:
            self._client._request("close")
        finally:
            self._closed = True
            self._client._session = None
