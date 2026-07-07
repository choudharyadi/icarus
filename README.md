# Icarus — AI Grand Prix Virtual Qualifier

Autonomous drone-racing stack for the **AI Grand Prix Virtual Qualifier**
(spec `VADR-TS-002`, see `documents/competition_technical_specs.txt`),
developed and raced entirely inside Webots against a faithful local replica
of the competition simulator's interface.

![Webots](https://img.shields.io/badge/Webots-R2025a-blue)
![Python](https://img.shields.io/badge/Python-3.12-green)
![MAVLink](https://img.shields.io/badge/MAVLink-2-orange)

```
┌─────────────────────────────┐         UDP          ┌─────────────────────────┐
│  Webots world               │                      │  pilot/  (the entry)    │
│  ┌───────────────────────┐  │  MAVLink 2 :14550    │  ┌───────────────────┐  │
│  │ dcl_sim_bridge        │──┼──────────────────────┼─▶│ comms             │  │
│  │  - flight controller  │  │  telemetry, race     │  │ state             │  │
│  │  - MAVLink server     │◀─┼──────────────────────┼──│ perception (PnP)  │  │
│  │  - race manager       │  │  setpoints, arm      │  │ planning          │  │
│  │  - vision streamer    │──┼──────────────────────┼─▶│ guidance          │  │
│  └───────────────────────┘  │  JPEG chunks :5600   │  └───────────────────┘  │
└─────────────────────────────┘                      └─────────────────────────┘
```

The pilot is **sim-agnostic**: it speaks only the interfaces defined in the
technical spec (MAVLink 2 telemetry/commands + the chunked JPEG vision
stream), so the same `pilot/` package runs unchanged against the real DCL
simulator — only speed-envelope retuning in `pilot/config.py` is expected.

## Results (Webots, real-time physics @120 Hz)

| Course | Gates | Time | Collisions |
|---|---|---|---|
| `worlds/Qualifier1.wbt` (11 gates, 80 m arc) | 11/11 | **39.6 s** | 0 |
| `worlds/Qualifier2.wbt` (12 gates, slalom + 2 m altitude swings) | 12/12 | **35.1 s** | 0 |

## Quick start

```bash
# 1. Install deps into the Python that Webots uses
#    (Webots > Preferences > General > Python command)
python3 -m pip install -r requirements.txt

# 2. Open worlds/Qualifier1.wbt in Webots and press Play.
#    The bridge auto-launches the pilot; console shows [BRIDGE]/[PILOT]/[RACE].

# 3. Or fully headless from a terminal:
tools/race.sh 1        # Qualifier 1
tools/race.sh 2        # Qualifier 2

# 4. Unit tests (frames, wire formats, detector, planner):
python3 -m pytest tests/ -q
```

To run the pilot manually (e.g. against the real simulator), disable
auto-launch and start it yourself:

```bash
ICARUS_AUTOPILOT=0  # set in the environment Webots inherits, then press Play
python3 pilot/main.py [--viz] [--save-frames]
```

## Repository layout

```
controllers/dcl_sim_bridge/    Webots controller = competition simulator replica
  dcl_sim_bridge.py            main loop: sensors -> telemetry, control in, race mgmt
  flight_controller.py         "stabilized controller": NED setpoints -> motors
  mavlink_server.py            MAVLink 2 UDP server (telemetry, commands, track data)
  vision_streamer.py           640x360 JPEG @30 Hz, chunked UDP per spec 4.6
  race_manager.py              gate discovery, passage detection, timing, collisions
  frames.py / wire.py          ENU<->NED math, binary payload formats
  pilot_launcher.py            auto-spawns pilot/main.py (ICARUS_AUTOPILOT=0 to disable)

pilot/                         THE COMPETITION ENTRY (portable, spec-only interfaces)
  main.py                      entry point + control loop (50 Hz)
  config.py                    every tunable in one place
  comms/mavlink_io.py          telemetry RX, heartbeat/timesync TX, setpoint TX
  comms/vision_rx.py           JPEG chunk reassembly -> latest-frame buffer
  perception/gate_detector.py  HSV red mask -> inner-quad PnP pose (spec intrinsics)
  planning/track.py            gate map from track data + vision corrections
  planning/racing_line.py      approach/center/exit waypoints, carrot, speed plan
  guidance.py                  INIT -> ARM -> TAKEOFF -> RACE -> FINISH state machine
  state.py / frames.py         thread-safe shared state, NED/FRD/camera math
  logging_util.py              per-run CSV/event/summary logs in runs/

worlds/Qualifier1.wbt          start + 9 intermediate + finish gates, obstacles
worlds/Qualifier2.wbt          harder: tighter spacing, sharper turns, altitude swings
protos/QualifierGate.proto     spec-dimension gate (2.7 m outer, 1.5 m inner, 0.26 m deep)
tests/                         31 unit tests (pytest)
tools/race.sh                  one-command headless qualifier run

controllers/main_controller/   legacy manual-flight / early prototype (kept for reference)
```

## Spec compliance (what the bridge replicates, what the pilot consumes)

| Spec item | Implementation |
|---|---|
| 3.2 Physics 120 Hz | `basicTimeStep 8` ms |
| 3.7 Gate 2.7 m / 1.5 m / 0.26 m | `protos/QualifierGate.proto` |
| 3.8 NED frames, camera 20° up, 640×360, fx=fy=320, cx,cy=(320,180) | world camera + `frames.py` both sides |
| 4.2/4.3 MAVLink 2 UDP: HEARTBEAT, ATTITUDE, HIGHRES_IMU, LOCAL_POSITION_NED, ODOMETRY, TIMESYNC, SET_POSITION_TARGET_LOCAL_NED, SET_ATTITUDE_TARGET, COLLISION, ENCAPSULATED_DATA | `mavlink_server.py` / `mavlink_io.py` |
| 4.4 Command < 100 Hz, heartbeat ≥ 2 Hz | pilot streams setpoints at 50 Hz, heartbeats at 4 Hz |
| 4.6 Vision stream: UDP 5600, 24 B header `<IHHIIQ`, chunked JPEG, 30 Hz | `vision_streamer.py` / `vision_rx.py` |
| Race status + track info `ENCAPSULATED_DATA` (formats from the official example client) | `wire.py`, round-trip unit tested |
| ARM/DISARM, SIM_RESET (31000) | `COMMAND_LONG` handling in the bridge |
| 8.3 Max run 8 min | bridge `ICARUS_MAX_SIM_T` watchdog |

The connection model matches the official PyAIPilotExample: the **client
listens** (`udpin 0.0.0.0:14550`, vision bind `:5600`) and the simulator
transmits to it.

## How the pilot races

1. **INIT** — waits for telemetry and the track description (gate poses in
   NED, transmitted by the sim as chunked `ENCAPSULATED_DATA`).
2. **ARM / TAKEOFF** — arms, climbs at the first gate's altitude, aims the
   nose (and therefore the camera) at gate 0.
3. **RACE** — for each gate the planner lays *approach → center → exit*
   waypoints along the gate normal. A carrot point (lookahead grows with
   planned speed) is chased with velocity+yaw setpoints. Speed is scheduled:
   cruise on straights, braking profiles into gates, gate speed shaped by
   how sharp the next heading change is. Yaw always points the camera at the
   next gate to be crossed.
4. **Perception in the loop** — every camera frame is HSV-masked for the
   red frame; the inner opening's four corners go through `solvePnP`
   (IPPE, planar square of known 1.5 m size) with hard rejection filters
   (reprojection error, range, aspect, opening ratio). Accepted detections
   are fused as bounded, smoothed per-gate corrections to the track map —
   so telemetry drift relative to what the camera actually sees is cancelled
   near every gate.
5. **FINISH** — crossing the finish gate stops the clock (race status from
   the sim is authoritative); the pilot holds position and writes
   `runs/run_*/summary.json`, a full CSV flight log and an event log.

## Tuning

Everything lives in `pilot/config.py`:

- `v_cruise / v_gate / decel` — the speed envelope. Current values finish
  Qualifier 1 in ~40 s with zero collisions; raise for more risk.
- `approach_dist / exit_dist` — how far out the gate-threading waypoints sit.
- `lookahead*` — carrot geometry (responsiveness vs corner-cutting).
- `correction_*` — vision-fusion aggressiveness.

Bridge-side flight-controller limits (tilt cap, velocity caps) are in
`controllers/dcl_sim_bridge/flight_controller.py`.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `ICARUS_AUTOPILOT` | `1` | bridge auto-launches `pilot/main.py` |
| `ICARUS_AUTOQUIT` | `0` | quit Webots when the race ends (batch/CI) |
| `ICARUS_MAX_SIM_T` | `480` | sim-time cap for auto-quit runs (s) |
| `ICARUS_CLIENT_IP` | `127.0.0.1` | where the bridge sends MAVLink + video |
| `ICARUS_VIZ` | `0` | pilot shows live annotated detection window |
| `ICARUS_SAVE_FRAMES` | `0` | pilot saves annotated frames to the run dir |
