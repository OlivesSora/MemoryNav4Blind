"""Length-prefixed pickle framing for the CAT-Seg worker socket.

The protocol keeps the binary frame/mask payloads on a dedicated Unix
domain socket so CAT-Seg/detectron2 logging on stdout/stderr cannot corrupt
the channel.
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any, Optional

HEADER = struct.Struct("!I")
MAX_MESSAGE_BYTES = 128 * 1024 * 1024


def send_message(connection: socket.socket, payload: dict[str, Any]) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
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
    return pickle.loads(data)
