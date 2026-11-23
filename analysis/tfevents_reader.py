"""Minimal, dependency-free TensorBoard event-file reader (scalars only; the TFRecord CRCs are not checked).

It exists so that the Stage-2 validation curves can be regenerated without TensorBoard. It was cross-checked
against ``tensorboard.backend.event_processing.event_accumulator`` (see docs/PROTOCOL.md, section 9).
"""
from __future__ import annotations

import struct
from typing import Dict, Iterator, List, Tuple


def _varint(buf: bytes, i: int) -> Tuple[int, int]:
    shift = 0
    result = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes) -> Iterator[Tuple[int, int, object]]:
    """Yield (field number, wire type, value) of one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 2:
            length, i = _varint(buf, i)
            value, i = buf[i:i + length], i + length
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def read_events(path: str) -> Iterator[Dict[str, object]]:
    """Yield ``{'wall_time', 'step', 'scalars': [(tag, value), ...]}`` for every record of an event file."""
    with open(path, "rb") as handle:
        data = handle.read()
    i, n = 0, len(data)
    while i + 12 <= n:
        (length,) = struct.unpack("<Q", data[i:i + 8])
        i += 12                                    # length (8) + length CRC (4)
        record = data[i:i + length]
        i += length + 4                            # payload + payload CRC (4)
        event: Dict[str, object] = {"wall_time": None, "step": 0, "scalars": []}
        for field, wire, value in _fields(record):
            if field == 1 and wire == 1:
                (event["wall_time"],) = struct.unpack("<d", value)  # type: ignore[arg-type]
            elif field == 2 and wire == 0:
                event["step"] = value
            elif field == 5 and wire == 2:         # tf.Summary
                for f2, w2, v2 in _fields(value):  # type: ignore[arg-type]
                    if f2 == 1 and w2 == 2:        # Summary.Value
                        tag, scalar = None, None
                        for f3, w3, v3 in _fields(v2):  # type: ignore[arg-type]
                            if f3 == 1 and w3 == 2:
                                tag = v3.decode("utf-8", "replace")  # type: ignore[union-attr]
                            elif f3 == 2 and w3 == 5:
                                (scalar,) = struct.unpack("<f", v3)  # type: ignore[arg-type]
                        if tag is not None and scalar is not None:
                            event["scalars"].append((tag, scalar))  # type: ignore[union-attr]
        yield event


def read_scalars(path: str) -> Tuple[Dict[str, List[Tuple[int, float, float]]], List[float]]:
    """Return ``({tag: [(step, value, wall_time), ...]}, [all wall times])`` of one event file."""
    scalars: Dict[str, List[Tuple[int, float, float]]] = {}
    wall_times: List[float] = []
    for event in read_events(path):
        if event["wall_time"] is not None:
            wall_times.append(event["wall_time"])  # type: ignore[arg-type]
        for tag, value in event["scalars"]:  # type: ignore[union-attr]
            scalars.setdefault(tag, []).append((event["step"], value, event["wall_time"]))  # type: ignore[arg-type]
    return scalars, wall_times
