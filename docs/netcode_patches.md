# Strike Eagle - Sandbox Patches

## Overview

The upstream HarFang3D Dogfight Sandbox has several critical bugs in its
network server that make it unsuitable for RL training, and lacks cockpit
audio features expected in a combat sim. We patched the source to fix these
issues, added new network commands for performance and camera control,
implemented a full RWR (Radar Warning Receiver) audio system, and added
Betty voice callouts.

All changes are in `refs/dogfight-sandbox-hg2/source/`.

## Bug Fixes

### 1. Silent Error Swallowing (network_server.py)

**Problem:** The server's main loop had a bare `except:` that caught ALL
exceptions — including real crashes — and printed the same generic
`"server_update ERROR"` message. This made debugging impossible. Clean
client disconnects were indistinguishable from server crashes.

**Before:**

```python
except:
print("network_server.py - server_update ERROR")
flag_server_connected = False
main.flag_client_update_mode = False
main.set_renderless_mode(False)
msg = "Socket closed"
print(msg)
```

**After:**

```python
# Clean disconnect detection
if answ is None:
    print("Client disconnected (clean)")
    break

# Per-command error isolation with full traceback
try:
    commands_functions[cmd_name](command["args"])
except Exception as e:
    error_msg = "ERROR executing command '{}': {}\n{}".format(
        cmd_name, e, traceback.format_exc())
    print(error_msg)
```

**Impact:** Failing commands no longer kill the connection. Clean disconnects
are properly identified. Real errors show the exact command, exception, and
full traceback.

### 2. get_missile_state Crash (network_server.py)

**Problem:** Calling `get_missile_state` on a recently-fired missile crashed
the server because the missile object had uninitialized attributes. The bare
`except:` in the main loop caught this and killed the connection.

**After:** The handler now:

1. Checks if the missile ID exists in `destroyables_items`
2. Wraps attribute access in try/except
3. Returns a safe fallback (`active: false`) on any error
4. Always sends a valid JSON response — never crashes

### 3. Unconditional Logging (network_server.py)

**Problem:** `get_planes_list`, `get_missiles_list`, and
`get_missile_launchers_list` always printed to console regardless of
`disable_log()`. During RL training with thousands of steps, this created
massive console spam.

**Fix:** All print statements gated behind `if flag_print_log:`.

### 4. Rendered-Mode Race Condition (network_server.py, master.py, main.py)

**Problem:** In rendered (non-renderless) mode, `update_scene()` just set a
flag (`flag_client_ask_update_scene = True`) and returned immediately. The
actual physics update ran on the main thread asynchronously. The network
thread continued processing commands while the main thread was mid-update,
causing concurrent modification of game state → crashes.

**Root cause in main.py:**

```python
# Main thread loop
if (not flag_client_update_mode) or ((not flag_renderless) and flag_client_ask_update_scene):
    Main.update()  # Physics + render runs here
else:
    time.sleep(1 / 120)
```

In renderless mode, `update_scene()` called `main.update()` directly on the
network thread (synchronous). In rendered mode, it set a flag for the main
thread (asynchronous). This asymmetry caused the race.

**Fix:** Added a `threading.Event` synchronization primitive:

- `master.py`: Added `client_update_done = threading.Event()` to Main class
- `network_server.py`: Rendered-mode `update_scene()` now clears the event,
  sets the flag, then **blocks** with `client_update_done.wait()`
- `main.py`: Main loop signals `client_update_done.set()` after `Main.update()`

This makes rendered mode synchronous like renderless mode — the network
thread cannot process the next command until the physics/render frame is done.

## New Commands

### STEP (network_server.py)

Combined command for RL training: applies controls, advances simulation, and
returns plane state in a single round-trip.

```json
{
  "command": "STEP",
  "args": {
    "plane_id": "ally_1",
    "pitch": 0.5,
    "roll": -0.3,
    "yaw": 0.0,
    "thrust": 1.0
  }
}
```

Returns the full plane state dict (same as `GET_PLANE_STATE`).

**Performance impact:** Reduces per-step network operations from ~11 to 1.

### SET_CAMERA_TRACK (network_server.py)

Sets the 3D camera to follow a specific aircraft and makes it the audio
listener for spatialized SFX.

```json
{
  "command": "SET_CAMERA_TRACK",
  "args": {
    "machine_id": "ally_1"
  }
}
```

### SET_TRACK_VIEW (network_server.py)

Sets the camera angle relative to the tracked aircraft.

```json
{
  "command": "SET_TRACK_VIEW",
  "args": {
    "view": "back"
  }
}
```

Options: `back`, `front`, `left`, `right`, `top`.

## Transport Optimizations

### TCP_NODELAY (socket_lib.py, both client and server)

Disabled Nagle's algorithm on both ends. Nagle buffers small TCP writes for
up to 40ms before sending. For our request/response pattern with small JSON
messages, this added massive latency.

```python
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
```

### SO_REUSEADDR (server socket_lib.py)

Enabled on the server socket to prevent "address already in use" errors when
restarting the sandbox quickly after a crash or disconnect.

### Bind to All Interfaces (server socket_lib.py)

The upstream server binds to the LAN IP only (`socket.gethostbyname(hostname)`),
forcing clients to connect via the network stack even on the same machine.
Changed to bind to `""` (all interfaces) so clients can use `127.0.0.1`
(localhost), eliminating NIC round-trip latency for local training.

### Batched Writes (client socket_lib.py)

Added `send_messages_batch()` that concatenates multiple framed messages into
a single `sendall()` call, reducing kernel syscall overhead for fire-and-forget
command sequences.

## Missile Guidance Fix (network_server.py)

**Problem:** `FIRE_MISSILE` only gave the missile a guidance target if the
aircraft's `TargettingDevice` had `target_locked = True`. In network mode,
the targeting device rarely achieves lock because there's no continuous
radar simulation — so missiles always launched unguided (flying straight).

**Fix:** `FIRE_MISSILE` now accepts an optional `target_id` parameter. When
provided, it force-sets `td.target_locked = True` before firing, guaranteeing
the missile gets proper guidance regardless of lock status.

## Cockpit Audio System (Machines.py)

### RWR (Radar Warning Receiver)

A full threat warning system added to `AircraftSFX`:

| Sound | File | Trigger |
|-------|------|---------|
| Contact beep | `rwr_contact.wav` | New threat detected |
| Tracking tone (loop) | `rwr_tracking.wav` | Enemy radar locked on aircraft |
| Missile warning (loop) | `missile_warning.wav` | Active missile targeting aircraft |
| Clear tone | `rwr_clear.wav` | Threat gone (missile missed or lock broken) |

The system uses a state machine (`none` → `tracking` → `missile`) with
proper transitions — when state changes, the old sound stops, a contact
beep plays, and the new looping tone starts. When threat clears, the clear
tone plays once.

### Betty Voice Callouts

Synthetic voice warnings added to `AircraftSFX.update_sfx()`:

| Sound | File | Trigger | Cooldown |
|-------|------|---------|----------|
| "PULL UP! PULL UP!" | `betty_pullup.wav` | Altitude < 300m while airborne | 4s |
| "ALTITUDE!" | `betty_altitude.wav` | Altitude < 800m while airborne | 5s |
| "BINGO!" | `betty_bingo.wav` | Thrust < 10% while airborne | 8s |
| "FLIGHT CONTROLS!" | `betty_flight_controls.wav` + `deedle_deedle.wav` | Health < 70% (battle damage) | 6s |

All callouts require `is_airborne` (speed > 50 m/s and `flag_landed` is
False) to prevent triggering on the carrier deck.

### Custom SFX Files

Located in `refs/dogfight-sandbox-hg2/source/assets/sfx/`:

| File | Source | Description |
|------|--------|-------------|
| `rwr_contact.wav` | Extracted from F/A-18 ALR-67 | Single contact beep |
| `rwr_tracking.wav` | Extracted from F/A-18 ALR-67 | Slow radar tracking tone |
| `missile_warning.wav` | Extracted from F/A-18 ALR-67 | Fast missile launch warning |
| `rwr_clear.wav` | Extracted from F/A-18 ALR-67 | Threat clear tone |
| `betty_altitude.wav` | Voice callout | "ALTITUDE!" |
| `betty_pullup.wav` | Voice callout | "PULL UP! PULL UP!" |
| `betty_bingo.wav` | Voice callout | "BINGO!" |
| `betty_flight_controls.wav` | Voice callout | "FLIGHT CONTROLS!" |
| `deedle_deedle.wav` | Warning tone | Deedle-deedle alert chime |

## Performance Summary

| Metric                            | Before patches          | After patches |
|-----------------------------------|-------------------------|---------------|
| Network ops per step              | ~11                     | 1             |
| Latency per message               | 1-40ms (Nagle)          | ~0.1ms        |
| Server crash on disconnect        | Always                  | Never         |
| Server crash on bad missile query | Always                  | Never         |
| Rendered mode stability           | Broken (race condition) | Stable        |
| Console spam with disable_log     | Yes                     | No            |
| Missile guidance in network mode  | Broken (always unguided)| Working       |
| Cockpit RWR audio                 | None                    | Full system   |
| Betty voice callouts              | None                    | 4 callouts    |
