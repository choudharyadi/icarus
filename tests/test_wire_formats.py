"""Wire-format round trips: what the bridge packs, the pilot must parse,
byte-for-byte compatible with the official example client's struct formats.
"""

import struct

import numpy as np

import wire  # controllers/dcl_sim_bridge/wire.py


def test_race_status_roundtrip():
    payload = wire.pack_race_status(123456, 1000, -1, 4, 7890)
    assert len(payload) == wire.ENCAPSULATED_DATA_LEN
    # Format string copied from the official example client.
    (data_type, sim_boot_ms, start_ms, finish_ns, active_gate,
     last_gate) = struct.unpack_from("<BQqqIq", payload)
    assert data_type == 1
    assert sim_boot_ms == 123456
    assert start_ms == 1000
    assert finish_ns == -1
    assert active_gate == 4
    assert last_gate == 7890


def _make_gates(n):
    return [{
        "id": i,
        "pos_ned": [i * 1.0, -i * 2.0, -2.5],
        "quat_ned": [1.0, 0.0, 0.0, 0.0],
        "width": 1.5,
        "height": 1.5,
    } for i in range(n)]


def _client_reassemble_and_parse(chunks):
    """Replicates the example client's logic exactly."""
    full = b""
    for chunk in chunks:
        data_type, transfer_id = struct.unpack_from("<BH", chunk)
        assert data_type == wire.TRACK_INFO_MSG_ID
        full += chunk[3:]
    num_gates, = struct.unpack_from("<H", full)
    full = full[2:]
    gates = []
    for _ in range(num_gates):
        rec = struct.unpack_from("<Hfffffffff", full)
        full = full[38:]
        gates.append(rec)
    return gates


def test_track_payload_single_chunk():
    gates = _make_gates(3)
    payload = wire.pack_track_payload(gates)
    chunks = wire.chunk_track_payload(payload, transfer_id=9)
    assert len(chunks) == 1
    assert all(len(c) == wire.ENCAPSULATED_DATA_LEN for c in chunks)
    parsed = _client_reassemble_and_parse(chunks)
    assert len(parsed) == 3
    for i, rec in enumerate(parsed):
        assert rec[0] == i
        assert np.isclose(rec[1], i * 1.0)
        assert np.isclose(rec[2], -i * 2.0)
        assert np.isclose(rec[3], -2.5)
        assert np.isclose(rec[8], 1.5)


def test_track_payload_multi_chunk():
    gates = _make_gates(20)  # 2 + 20*38 = 762 bytes -> 4 chunks of <=250
    payload = wire.pack_track_payload(gates)
    chunks = wire.chunk_track_payload(payload, transfer_id=3)
    assert len(chunks) > 1
    parsed = _client_reassemble_and_parse(chunks)
    assert len(parsed) == 20
    assert [rec[0] for rec in parsed] == list(range(20))


def test_vision_header_roundtrip():
    hdr = wire.pack_vision_header(42, 3, 7, 99999, 1400, 123456789)
    assert len(hdr) == wire.VISION_HEADER_SIZE == 24
    frame_id, chunk_id, total, jpeg_size, payload_size, t_ns = struct.unpack(
        "<IHHIIQ", hdr)
    assert (frame_id, chunk_id, total, jpeg_size, payload_size, t_ns) == \
        (42, 3, 7, 99999, 1400, 123456789)
