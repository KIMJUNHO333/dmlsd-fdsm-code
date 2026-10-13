import argparse
from dataclasses import asdict
import math
from pathlib import Path
import socket
import sys
import traceback

from .adapters.duplex import load_duplex_model
from .audio import AudioSpec
from ._duplex_protocol import DuplexProtocolError, receive_message, send_message, validate_output


def _serve_connection(stream, model):
    session = None
    try:
        while True:
            try:
                request, audio = receive_message(stream)
            except (EOFError, ConnectionError, TimeoutError):
                break
            operation = request.get("op")
            if operation != "tick" and audio is not None:
                raise DuplexProtocolError("unexpected request audio")
            response = {"op": operation, "ok": True}
            output_audio = None
            if operation == "info":
                response["provenance"] = asdict(model.provenance)
            elif operation == "open":
                if session is not None:
                    raise RuntimeError("a duplex session is already open")
                session = model.open_session()
                if not isinstance(session.audio_spec, AudioSpec):
                    raise TypeError("session.audio_spec must be an AudioSpec")
                response["audio_spec"] = asdict(session.audio_spec)
            elif operation == "tick":
                if session is None:
                    raise RuntimeError("no duplex session is open")
                session.audio_spec.validate_input(audio)
                output = session.process_tick(audio)
                validate_output(output, session.audio_spec)
                response.update(native_text=output.native_text, generation_unit_id=output.generation_unit_id,
                                measurements=None if output.measurements is None else asdict(output.measurements))
                output_audio = output.audio
            elif operation == "close":
                if session is None:
                    raise RuntimeError("no duplex session is open")
                session.close()
                session = None
            else:
                raise DuplexProtocolError("unknown duplex operation")
            send_message(stream, response, output_audio)
    except Exception as exc:
        traceback.print_exc()
        try:
            send_message(stream, {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}})
        except (OSError, ValueError, DuplexProtocolError):
            pass
    finally:
        if session is not None:
            session.close()


def serve(model, *, host="127.0.0.1", port=8765, idle_timeout_s=600.0):
    if not math.isfinite(idle_timeout_s) or idle_timeout_s <= 0:
        raise ValueError("idle_timeout_s must be positive and finite")
    with socket.create_server((host, port)) as listener:
        print(f"Duplex server ready at {host}:{listener.getsockname()[1]}", flush=True)
        while True:
            stream, _ = listener.accept()
            with stream:
                stream.settimeout(idle_timeout_s)
                stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                _serve_connection(stream, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--idle-timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    if args.adapter_dir is not None:
        directory = args.adapter_dir.expanduser().resolve()
        if not directory.is_dir():
            parser.error("--adapter-dir is not an existing directory")
        sys.path.insert(0, str(directory))
    model = load_duplex_model(args.config)
    try:
        serve(model, host=args.host, port=args.port, idle_timeout_s=args.idle_timeout_s)
    except KeyboardInterrupt:
        print("Duplex server stopped", file=sys.stderr)
    finally:
        model.close()


if __name__ == "__main__":
    main()
