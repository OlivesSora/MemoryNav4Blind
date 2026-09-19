"""Length-prefixed framing for the CAT-Seg worker socket.

The protocol keeps the binary frame/mask payloads on a dedicated Unix
domain socket so CAT-Seg/detectron2 logging on stdout/stderr cannot corrupt
the channel.

The navigation process and the worker may run in different conda
environments with different NumPy major versions (for example NumPy 2 in
``blind`` and NumPy 1 in the detectron2 environment).  Pickling an ndarray
across majors fails because NumPy 2 pickles reference ``numpy._core``, which
does not exist in NumPy 1.  Payloads are therefore normalised to builtin
types (arrays become dtype/shape/bytes) before pickling, and rebuilt with
``numpy.frombuffer`` on receipt.
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any, Optional

import numpy as np

HEADER = struct.Struct("!I")
MAX_MESSAGE_BYTES = 128 * 1024 * 1024


def _encode_arrays(value: Any) -> Any:
    """Replace NumPy objects with version-independent builtin containers."""
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": contiguous.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": True, "items": [_encode_arrays(item) for item in value]}
    if isinstance(value, list):
        return [_encode_arrays(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_arrays(item) for key, item in value.items()}
    return value


def _decode_arrays(value: Any) -> Any:
    """Rebuild NumPy arrays encoded by :func:`_encode_arrays`."""
    if isinstance(value, list):
        return [_decode_arrays(item) for item in value]
    if isinstance(value, dict):
        if value.get("__ndarray__"):
            array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(tuple(value["shape"])).copy()
        if value.get("__tuple__"):
            return tuple(_decode_arrays(item) for item in value["items"])
        return {key: _decode_arrays(item) for key, item in value.items()}
    return value


def send_message(connection: socket.socket, payload: dict[str, Any]) -> None:
    data = pickle.dumps(_encode_arrays(payload), protocol=pickle.HIGHEST_PROTOCOL)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message too large: {len(data)} bytes")
    connection.sendall(HEADER.pack(len(data)) + data)


def _recv_exact(connection: socket.socket, count: int) -> Optional[bytes]:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = connection.recv(min(remaining, 1 << 20))
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(connection: socket.socket) -> Optional[dict[str, Any]]:
    header = _recv_exact(connection, HEADER.size)
    if header is None:
        return None
    (length,) = HEADER.unpack(header)
    if length > MAX_MESSAGE_BYTES:
        raise ValueError(f"message too large: {length} bytes")
    data = _recv_exact(connection, length)
    if data is None:
        return None
    return _decode_arrays(pickle.loads(data))
