"""Binary payload formats used inside MAVLink ENCAPSULATED_DATA messages and
the UDP vision stream, matching the AI Grand Prix technical specification
(VADR-TS-002) and the official PyAIPilotExample client.
"""

import struct

# ENCAPSULATED_DATA sub-message IDs
RACE_STATUS_MSG_ID = 1
TRACK_INFO_MSG_ID = 2

# Custom command issued by clients to reset the simulation
MAVLINK_CMD_SIM_RESET = 31000

# Collision classes
COLLISION_ID_GATE = 1001
COLLISION_ID_ENVIRONMENT = 1002

ENCAPSULATED_DATA_LEN = 253  # fixed payload size of ENCAPSULATED_DATA

# Race status payload (after example client: "<BQqqIq")
#   data_type, sim_boot_time_ms, race_start_boot_time_ms,
#   race_finish_time_ns, active_gate_index, last_gate_race_time_ms
RACE_STATUS_FMT = "<BQqqIq"

# Track info: header per chunk "<BH" (data_type, transfer_id), payload begins
# with "<H" num_gates then per-gate records:
#   gate_id, pos NED x/y/z, quat NED w/x/y/z, width, height
TRACK_GATE_FMT = "<Hfffffffff"
TRACK_GATE_SIZE = struct.calcsize(TRACK_GATE_FMT)  # 38 bytes

# Vision stream packet header (UDP port 5600)
VISION_HEADER_FMT = "<IHHIIQ"
VISION_HEADER_SIZE = struct.calcsize(VISION_HEADER_FMT)  # 24 bytes
VISION_CHUNK_PAYLOAD = 1400  # keep packets under typical MTU, like the real sim


def pack_race_status(sim_boot_time_ms, race_start_boot_time_ms,
                     race_finish_time_ns, active_gate_index,
                     last_gate_race_time_ms):
    payload = struct.pack(
        RACE_STATUS_FMT,
        RACE_STATUS_MSG_ID,
        int(sim_boot_time_ms),
        int(race_start_boot_time_ms),
        int(race_finish_time_ns),
        int(active_gate_index),
        int(last_gate_race_time_ms),
    )
    return payload.ljust(ENCAPSULATED_DATA_LEN, b"\x00")


def pack_track_payload(gates):
    """gates: iterable of dicts with keys
    id, pos_ned (3,), quat_ned (w,x,y,z), width, height."""
    out = struct.pack("<H", len(gates))
    for g in gates:
        out += struct.pack(
            TRACK_GATE_FMT,
            int(g["id"]),
            float(g["pos_ned"][0]), float(g["pos_ned"][1]), float(g["pos_ned"][2]),
            float(g["quat_ned"][0]), float(g["quat_ned"][1]),
            float(g["quat_ned"][2]), float(g["quat_ned"][3]),
            float(g["width"]), float(g["height"]),
        )
    return out


def chunk_track_payload(payload, transfer_id):
    """Split a full track payload into ENCAPSULATED_DATA-sized chunks.

    Each chunk: data_type(B) + transfer_id(H) + slice, padded to 253 bytes.
    Returns list of byte strings (one per chunk, in seqnr order).
    """
    header = struct.pack("<BH", TRACK_INFO_MSG_ID, transfer_id)
    max_slice = ENCAPSULATED_DATA_LEN - len(header)
    chunks = []
    for off in range(0, len(payload), max_slice):
        body = header + payload[off:off + max_slice]
        chunks.append(body.ljust(ENCAPSULATED_DATA_LEN, b"\x00"))
    return chunks


def pack_vision_header(frame_id, chunk_id, total_chunks, jpeg_size,
                       payload_size, sim_time_ns):
    return struct.pack(
        VISION_HEADER_FMT,
        frame_id & 0xFFFFFFFF,
        chunk_id,
        total_chunks,
        jpeg_size,
        payload_size,
        sim_time_ns,
    )
