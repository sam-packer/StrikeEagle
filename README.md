# Strike Eagle

A reinforcement learning agent that learns to evade incoming missiles in
a 3D air combat simulator. An F-16 Fighting Falcon controlled by a SAC
(Soft Actor-Critic) policy learns defensive maneuvering against guided
missiles fired by an enemy Eurofighter Typhoon.

Built on a patched fork of the
[HarFang3D Dogfight Sandbox](https://github.com/harfang3d/dogfight-sandbox-hg2)
with significant network server improvements for RL training stability
and performance.

## Prerequisites

- **Windows 10/11** with a GPU (tested on RTX 5090)
- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** package manager

## Quick Start

```
# 1. Install dependencies (one-time)
uv sync
cd refs/dogfight-sandbox-hg2 && python -m pip install -r source/requirements.txt && cd ../..

# 2. Launch the Dogfight Sandbox (see detailed instructions below)

# 3. Run the diagnostic to verify everything works
uv run diagnose

# 4. Train the agent
uv run train --algo sac --total-timesteps 500000

# 5. Watch the trained agent in 3D
uv run demo --model checkpoints/missile_evasion_sac_final.zip
```

---

## Step 1: Install the Dogfight Sandbox

This project includes a patched version of the Dogfight Sandbox at
`refs/dogfight-sandbox-hg2/` with fixes for network server stability
(proper error handling, crash-safe missile state queries, and correct
log gating). The upstream pre-built binaries do not include these fixes.

### Install sandbox dependencies (one-time):

```
cd refs/dogfight-sandbox-hg2
python -m pip install -r source/requirements.txt
```

This installs `harfang==3.2.7` and `tqdm`. No other dependencies are needed.

### Configure aircraft (optional):

Edit `refs/dogfight-sandbox-hg2/source/scripts/network_mission_config.json`
to control which aircraft spawn. The default creates 2 allies and 2 enemies.
For faster training, use one per side:

```json
{
  "aircrafts_allies": [
    "F16"
  ],
  "aircrafts_ennemies": [
    "Eurofighter"
  ],
  "aircraft_carriers_allies_count": 1,
  "aircraft_carriers_enemies_count": 1
}
```

This creates `ally_1` (F16, your agent) and `ennemy_1` (Eurofighter, the
missile launcher).

### Display resolution:

The sandbox renders at whatever resolution is set in
`refs/dogfight-sandbox-hg2/config.json`. The included config is set to
`3840x2160` (4K). If your display is a different resolution, update the
`"Resolution"` field to match your native resolution — otherwise you'll
see black space around the rendered area or a cropped window.

---

## Step 2: Launch the Sandbox in Network Mode

```
cd refs/dogfight-sandbox-hg2
1-start.bat
```

Then in the 3D window:

1. Use **LEFT/RIGHT arrow keys** to browse missions
2. Navigate to **"Network mode"** (first mission in the list — it starts on the second)
3. Press **SPACE** to start
4. The top-left corner will show:
   ```
   Hostname: YOUR-PC, IP: 192.168.x.x, port: 50888
   ```
   This confirms the network server is listening.

**Custom port:**

```
cd refs/dogfight-sandbox-hg2/source
python main.py network_port 12345
```

> **Tip:** The default port is **50888**. All scripts auto-detect your LAN IP
> (matching what the sandbox displays). You can override with `--host <ip>` if needed.

---

## Step 3: Verify the Connection

With the sandbox running in Network mode, open a **new terminal** and run:

```
uv run diagnose
```

This script will:

1. Connect to the sandbox
2. List all available aircraft and their IDs
3. Query aircraft state (position, speed, health)
4. Check missile systems
5. Test flight controls
6. Run 2 short episodes with random actions

If everything works, you'll see `ALL DIAGNOSTICS PASSED`.

**Troubleshooting the connection:**

| Problem                  | Fix                                                            |
|--------------------------|----------------------------------------------------------------|
| "Connection refused"     | Sandbox isn't running or isn't in Network mode                 |
| Hangs on "Connecting..." | Check firewall; try `--host <your-lan-ip>` from sandbox window |
| Wrong plane IDs          | Check `network_mission_config.json` and restart sandbox        |
| No missiles on aircraft  | All default aircraft models have missiles                      |

---

## Step 4: Train the Agent

```
uv run train --algo sac --total-timesteps 500000
```

### Training Options

| Flag                | Default        | Description                            |
|---------------------|----------------|----------------------------------------|
| `--algo`            | `sac`          | Algorithm: `sac` or `ppo`              |
| `--total-timesteps` | `500000`       | Total training steps                   |
| `--host`            | auto-detected  | Sandbox server host                    |
| `--port`            | `50888`        | Sandbox server port                    |
| `--max-steps`       | `1500`         | Max steps per episode                  |
| `--checkpoint-freq` | `10000`        | Save a checkpoint every N steps        |
| `--log-dir`         | `logs/`        | TensorBoard log directory              |
| `--save-dir`        | `checkpoints/` | Model checkpoint directory             |
| `--resume`          | -              | Path to `.zip` to resume training from |

### Monitor Training with TensorBoard

In a separate terminal:

```
uv run tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

Key metrics to watch:

- **ep_rew_mean**: average episode reward (should trend upward)
- **ep_len_mean**: average episode length (longer = surviving longer)

### Training Tips

- **SAC** (Soft Actor-Critic) is recommended for continuous control — it's more
  sample-efficient than PPO and handles exploration well.
- **Start with 200k-500k timesteps** to see if the agent learns basic evasion.
  A full training run might need 1M+ steps.
- Training runs in **renderless mode** by default for maximum speed. The sandbox
  window will go dark/minimal — this is normal.
- Each episode: the enemy fires a missile after ~10 steps, and the agent must
  evade. Episode ends on hit, successful evasion, crash, or timeout.
- **Resume training** if you need to stop and restart:
  ```
  uv run train --resume checkpoints/missile_evasion_10000_steps.zip
  ```

---

## Step 5: Watch the Trained Agent (Demo)

```
uv run demo --model checkpoints/missile_evasion_sac_final.zip
```

This runs with **full 3D rendering** enabled in the sandbox. You can use the
sandbox's camera controls to watch from different angles:

- **F2**: Cycle camera views (cockpit, chase, external)
- **Arrow keys**: Pan camera
- The demo runs 5 episodes by default (`--episodes N` to change)

This is what you'd record for a video demo.

### Demo Options

| Flag         | Default       | Description                                |
|--------------|---------------|--------------------------------------------|
| `--model`    | (required)    | Path to trained model `.zip`               |
| `--algo`     | `sac`         | Must match the algorithm used for training |
| `--episodes` | `5`           | Number of demo episodes                    |
| `--host`     | auto-detected | Sandbox host                               |
| `--port`     | `50888`       | Sandbox port                               |

---

## Project Structure

```
src/
  socket_lib.py              TCP socket layer with TCP_NODELAY + batched writes
  dogfight_client.py         HarFang3D network API wrapper + custom STEP command
  missile_evasion_env.py     Gymnasium environment for missile evasion
  net_utils.py               LAN IP auto-detection
train.py                     SB3 training script (PPO/SAC)
demo.py                      Evaluation with 3D rendering + camera tracking
diagnose.py                  Connection diagnostic + missile attack demo
docs/
  architecture.md            System architecture and data flow
  netcode_patches.md         All sandbox patches documented
  environment.md             Observation/action/reward design
  missiles_and_aircraft.md   Reference guide for sim aircraft and missiles
  training_guide.md          Training commands, monitoring, tips
test_env_offline.py          Offline smoke test (no sandbox needed)
refs/
  dogfight-sandbox-hg2/      Patched sandbox (network server fixes)
  HIRL4UCAV/                 Reference RL project (studied for API patterns)
```

### Rendered vs Renderless Mode

The sandbox supports two modes for network clients:

- **Renderless mode** (`set_renderless_mode(True)`) — the sandbox window goes
  dark and physics runs as fast as possible. Best for training throughput.
  `update_scene()` is synchronous on the network thread.
- **Rendered mode** (default) — full 3D rendering with cockpit views, terrain,
  and missile trails. Best for diagnostics, demos, and recording video.
  `update_scene()` is synchronized between the network and render threads via
  our threading patch (see below).

The `train.py` script uses renderless mode by default for speed. The `diagnose`
and `demo` scripts use rendered mode so you can watch.

### Sandbox Patches

We applied the following fixes to the sandbox source
(`refs/dogfight-sandbox-hg2/source/`):

**Network stability** (`network_server.py`, `master.py`, `main.py`):

- Proper disconnect handling with clean/error distinction
- Per-command error isolation with full tracebacks
- Crash-safe `get_missile_state` for fired missiles
- Log gating for `get_planes_list`, `get_missiles_list`, etc.
- Thread-safe rendered-mode `update_scene` via `threading.Event`
- Missile guidance fix: `FIRE_MISSILE` accepts `target_id` to force lock

**New network commands** (`network_server.py`):

- `STEP` — apply controls + advance sim + return state in one round-trip
- `SET_CAMERA_TRACK` — camera follows aircraft with audio listener
- `SET_TRACK_VIEW` — camera angle (back, front, left, right, top)

**Cockpit audio system** (`Machines.py`):

- RWR (Radar Warning Receiver): contact beep, tracking tone, missile warning,
  clear tone — F/A-18 ALR-67 style sounds with state machine transitions
- Betty voice callouts: "PULL UP!", "ALTITUDE!", "BINGO!",
  "FLIGHT CONTROLS!" with airborne-only gating and cooldown timers

**Transport** (`socket_lib.py` client + server):

- TCP_NODELAY on both ends, SO_REUSEADDR on server, batched writes

See `docs/netcode_patches.md` for full details with before/after code.

## Environment Design

**Observation (19 dimensions):**

- Agent position (x, y, z) — normalised by 10,000m
- Agent velocity vector (vx, vy, vz) — normalised by 800 m/s
- Agent Euler angles (roll, pitch, yaw) — normalised by pi
- Agent speed (scalar) — normalised by 800 m/s
- Agent altitude — normalised by 10,000m
- Missile tracking (8 dims) — reserved, currently zeros due to sandbox
  `get_missile_state` limitations; evasion detected via health monitoring

**Action (4 dimensions, continuous):**

- Pitch rate: [-1, 1]
- Roll rate: [-1, 1]
- Yaw rate: [-1, 1]
- Thrust: [0, 1]

**Reward Function:**

| Condition                           | Reward                             |
|-------------------------------------|------------------------------------|
| Per timestep alive                  | +1.0                               |
| Missile evaded (survived 500 steps) | +100.0                             |
| Survived to max steps               | +50.0                              |
| Hit by missile                      | -100.0                             |
| Crashed into ground                 | -100.0                             |
| Extreme manoeuvres penalty          | -0.05 * (pitch^2 + roll^2 + yaw^2) |
| Low altitude (< 1000m)              | -1.0 to -5.0                       |
| High altitude (> 9000m)             | -1.0 to -5.0                       |

---

## Troubleshooting

### Sandbox won't start

- Make sure harfang is installed: `python -m pip install harfang==3.2.7`
- Check `config.json` — set `"OpenGL": true` on non-VR systems.

### Training is very slow

- Confirm renderless mode is active (the sandbox window should look dark/minimal).
- Close other GPU-heavy applications.
- Reduce `--max-steps` for shorter episodes during initial experimentation.

### Agent doesn't learn / reward stays flat

- Check TensorBoard — if `ep_len_mean` is always 10-20, the missile is hitting
  immediately. Try increasing `missile_fire_delay` in the env constructor.
- Try PPO if SAC isn't converging: `--algo ppo`
- Increase total timesteps — RL in physics sims can be slow to learn.

### "Connection reset" errors during training

- The sandbox may have crashed or the mission ended. Restart the sandbox in
  Network mode and re-run training with `--resume`.
