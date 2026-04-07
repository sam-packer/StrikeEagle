# Strike Eagle - Netcode Patches

## Overview

The upstream HarFang3D Dogfight Sandbox has several critical bugs in its
network server that make it unsuitable for RL training. We patched the source
to fix these issues, added new commands for performance and camera control,
and optimized the transport layer.

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

### Batched Writes (client socket_lib.py)

Added `send_messages_batch()` that concatenates multiple framed messages into
a single `sendall()` call, reducing kernel syscall overhead for fire-and-forget
command sequences.

## Performance Summary

| Metric                            | Before patches          | After patches |
|-----------------------------------|-------------------------|---------------|
| Network ops per step              | ~11                     | 1             |
| Latency per message               | 1-40ms (Nagle)          | ~0.1ms        |
| Server crash on disconnect        | Always                  | Never         |
| Server crash on bad missile query | Always                  | Never         |
| Rendered mode stability           | Broken (race condition) | Stable        |
| Console spam with disable_log     | Yes                     | No            |
