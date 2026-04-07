# Strike Eagle - Environment Design

## Missile Evasion Gymnasium Environment

`src/missile_evasion_env.py` implements a Gymnasium-compatible environment
that wraps the HarFang3D Dogfight Sandbox for RL training.

## Observation Space (19 dimensions, float32)

| Index | Feature                         | Normalization | Description                      |
|-------|---------------------------------|---------------|----------------------------------|
| 0-2   | Position (x, y, z)              | / 10,000      | Agent world position in metres   |
| 3-5   | Velocity (vx, vy, vz)           | / 800         | Agent velocity vector in m/s     |
| 6-8   | Euler angles (roll, pitch, yaw) | / pi          | Agent orientation in radians     |
| 9     | Speed (scalar)                  | / 800         | Total linear speed in m/s        |
| 10    | Altitude                        | / 10,000      | Height above sea level in metres |
| 11-17 | Missile tracking (reserved)     | various       | Currently zeros (see note)       |
| 18    | Missile closing rate (reserved) | / 2,000       | Currently zero                   |

**Note on missile observations:** The upstream sandbox's `get_missile_state`
command crashes the server when querying recently-fired missiles (the missile
object has uninitialized attributes). Our server patch returns a safe fallback,
but the missile state is unreliable enough that we zero these dimensions and
detect evasion via health monitoring instead.

## Action Space (4 dimensions, float32)

| Index | Control | Range   | Description          |
|-------|---------|---------|----------------------|
| 0     | Pitch   | [-1, 1] | Nose up/down rate    |
| 1     | Roll    | [-1, 1] | Bank left/right rate |
| 2     | Yaw     | [-1, 1] | Rudder left/right    |
| 3     | Thrust  | [0, 1]  | Engine throttle      |

## Reward Function

The reward is computed per step from the cached plane state and health:

| Condition                                | Reward                    | Purpose                          |
|------------------------------------------|---------------------------|----------------------------------|
| Each timestep alive                      | +1.0                      | Incentivize survival             |
| Missile evaded (250 steps post-fire)     | +100.0                    | Big bonus for successful evasion |
| Survived to max steps                    | +50.0                     | Partial credit for endurance     |
| Hit by missile (health drops)            | -100.0                    | Strong penalty for failure       |
| Crashed (altitude < 50m or crashed flag) | -100.0                    | Don't fly into the ground        |
| Extreme manoeuvres                       | -0.05 * (p^2 + r^2 + y^2) | Discourage random flailing       |
| Low altitude (< 500m)                    | -5.0                      | Hard floor penalty               |
| Low altitude (500-1000m)                 | -1.0                      | Soft floor penalty               |
| High altitude (> 10,000m)                | -5.0                      | Hard ceiling penalty             |
| High altitude (9,000-10,000m)            | -1.0                      | Soft ceiling penalty             |

## Episode Setup

Each episode configures the scenario:

1. **Agent (F16):** Spawns at ~4000m altitude with slight random offset,
   flying at 300 m/s, thrust at 100%, gear retracted.

2. **Enemy (Eurofighter):** Spawns 2km behind the agent at the same altitude,
   flying at 300 m/s. Controls set to fly straight (no maneuvering).

3. **Targeting:** Enemy's targeting device is locked onto the agent via
   `set_target_id`.

4. **Missiles:** Enemy is rearmed with full missile loadout. After 30 steps
   (~1 second of sim time), the enemy fires from the first available slot.
   The missile's guidance target is explicitly set to the agent via
   `set_missile_target`.

## Termination Conditions

| Condition      | Type       | When                                     |
|----------------|------------|------------------------------------------|
| Health dropped | terminated | `health_level < 0.99`                    |
| Crashed        | terminated | `crashed` flag or altitude < 50m         |
| Missile evaded | terminated | 250 steps since fire, still alive        |
| Max steps      | truncated  | `step_count >= max_steps` (default 1500) |

## Evasion Detection

Since we cannot reliably query missile state, evasion is detected by
survival time: if the agent survives 250 steps (~8.3 seconds at 1/30s
timestep) after the missile fires without taking damage, the missile is
considered to have missed.

This is conservative — most missiles in the sim have fuel for 5-10 seconds
of flight. An agent that survives 8+ seconds after launch has almost
certainly evaded.

## Sim Timestep

The sandbox's physics timestep is set to 1/30s (double the default 1/60s).
This means each `step()` call covers more simulated time, reducing the
total number of steps needed for the same scenario duration and improving
training throughput.

## Network Efficiency

Each `step()` call uses our custom `STEP` server command: one TCP round-trip
that applies controls, advances physics, and returns the new plane state.
This is the minimum possible network overhead — equivalent to a local
function call in terms of data flow.
