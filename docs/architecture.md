# Strike Eagle - Architecture

## System Overview

Strike Eagle trains a reinforcement learning agent to evade incoming missiles
in a 3D air combat simulator. The system has three layers:

```
+---------------------+     TCP/JSON      +------------------------+
|   RL Training Loop  | <--------------> |  HarFang3D Dogfight    |
|   (Python / SB3)    |   port 50888     |  Sandbox (patched)     |
+---------------------+                  +------------------------+
| Gymnasium Env       |                  | Physics Engine         |
| SAC / PPO Agent     |                  | Aircraft Models        |
| TensorBoard Logs    |                  | Missile Simulation     |
+---------------------+                  | 3D Rendering           |
                                         +------------------------+
```

### Layer 1: Dogfight Sandbox (Simulation Server)

The HarFang3D Dogfight Sandbox is an open-source 3D air combat simulator. It
runs as a server, accepting commands over TCP sockets using a JSON-based
protocol. We run a patched fork with significant network server improvements
(see `docs/netcode_patches.md`).

The sandbox handles:

- Aircraft physics (lift, drag, thrust, gravity)
- Missile guidance and tracking (patched for network mode reliability)
- Collision detection and damage
- 3D rendering (when not in renderless mode)
- Sound effects (engine, explosions, wind)
- RWR threat warning system (custom: contact, tracking, missile, clear tones)
- Betty voice callouts (custom: PULL UP, ALTITUDE, BINGO, FLIGHT CONTROLS)

### Layer 2: Network Client (src/)

Our client library communicates with the sandbox:

- **`socket_lib.py`** - TCP socket layer with `TCP_NODELAY`, batched writes
- **`dogfight_client.py`** - ~50 API wrappers + custom `STEP`/`STEP_N` commands
- **`net_utils.py`** - Default host configuration (localhost)
- **`wrappers.py`** - `ActionRepeat` wrapper for frame skipping via `STEP_N`

The critical optimizations are the `STEP` and `STEP_N` commands: custom
server-side commands that apply controls, advance physics (N frames for
`STEP_N`), and return aircraft state in a single TCP round-trip. Without
these, each env step required ~11 network operations. With them: 1.

### Layer 3: RL Training (Gymnasium + Stable-Baselines3)

- **`missile_evasion_env.py`** - Gymnasium wrapper that translates between
  the RL agent's observations/actions and the sandbox's network API
- **`train.py`** - SAC/PPO training with TensorBoard logging and checkpoints
- **`demo.py`** - Runs trained policy with 3D rendering and camera tracking

## Data Flow Per Step

```
Agent                    Env                      Sandbox Server
  |                       |                            |
  |-- action[4] --------->|                            |
  |                       |-- STEP_N(pitch,roll,yaw,   |
  |                       |   thrust, n=4) ---------->|
  |                       |                            |-- apply controls
  |                       |                            |-- main.update() x N
  |                       |                            |   (physics + render)
  |                       |<-- plane_state dict -------|
  |                       |                            |
  |                       |-- GET_MISSILE_STATE ------>|
  |                       |<-- missile_state dict -----|  (per tracked missile)
  |                       |                            |
  |                       |-- build obs[27]            |
  |                       |-- compute reward           |
  |                       |-- check termination        |
  |                       |                            |
  |<-- obs, reward, done -|                            |
```

## Episode Lifecycle

1. **Reset**: Teleport both aircraft to starting positions. Enemy 2km behind
   the agent at 4000m altitude. Set target lock. Rearm missiles.

2. **Pre-fire phase** (steps 1-29): Agent flies, enemy chases. Targeting
   system acquires lock. Agent learns basic flight.

3. **Missile fire** (step 30): Enemy fires a missile. Missile target is
   explicitly set to the agent via `set_missile_target`.

4. **Evasion phase** (steps 31+): Agent must maneuver to survive. Missile
   tracks the agent with onboard guidance.

5. **Termination**: Episode ends when:
    - Agent health drops (missile hit) → reward -100
    - Agent crashes (altitude < 50m) → reward -100
    - All tracked missiles destroyed/deactivated → reward +100 (evaded)
    - Max steps reached → reward +50

## Training Configuration

| Parameter        | Value      | Rationale                                  |
|------------------|------------|--------------------------------------------|
| Algorithm        | SAC        | Best for continuous control                |
| Network          | [512, 512] | Larger network for complex maneuvers       |
| Batch size       | 1024       | Maximize GPU utilization                   |
| Gradient steps   | 4          | More learning per expensive env step       |
| Buffer size      | 500k       | Diverse replay for large batches           |
| Sim timestep     | 1/30s      | 2x faster than default, fewer steps needed |
| Discount (gamma) | 0.99       | Long-horizon survival matters              |
