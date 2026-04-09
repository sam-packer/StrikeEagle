# Strike Eagle - Environment Design

## Missile Evasion Gymnasium Environment

`src/missile_evasion_env.py` implements a Gymnasium-compatible environment
that wraps the HarFang3D Dogfight Sandbox for RL training.

## Observation Space (27 dimensions, float32)

| Index | Feature                         | Normalization | Description                           |
|-------|---------------------------------|---------------|---------------------------------------|
| 0-2   | Position (x, y, z)              | / 10,000      | Agent world position in metres        |
| 3-5   | Velocity (vx, vy, vz)           | / 800         | Agent velocity vector in m/s          |
| 6-8   | Euler angles (roll, pitch, yaw) | / pi          | Agent orientation in radians          |
| 9     | Speed (scalar)                  | / 800         | Total linear speed in m/s             |
| 10    | Altitude                        | / 10,000      | Height above sea level in metres      |
| 11-13 | Missile 1 relative position     | / 20,000      | (missile_pos - agent_pos) vector      |
| 14-16 | Missile 1 relative velocity     | / 2,000       | (missile_vel - agent_vel) vector      |
| 17    | Missile 1 distance              | / 20,000      | Euclidean distance to missile         |
| 18    | Missile 1 closing rate          | / 2,000       | Rate of distance change (+ = closing) |
| 19-21 | Missile 2 relative position     | / 20,000      | Same as above for SAM missile         |
| 22-24 | Missile 2 relative velocity     | / 2,000       | Same as above for SAM missile         |
| 25    | Missile 2 distance              | / 20,000      | Same as above for SAM missile         |
| 26    | Missile 2 closing rate          | / 2,000       | Same as above for SAM missile         |

Missile observations are populated from live `get_missile_state` queries each
step. Slots for missiles that haven't been fired yet or are already
destroyed/deactivated are filled with zeros.

## Action Space (5 dimensions, float32)

| Index | Control | Range   | Description          |
|-------|---------|---------|----------------------|
| 0     | Pitch   | [-1, 1] | Nose up/down rate    |
| 1     | Roll    | [-1, 1] | Bank left/right rate |
| 2     | Yaw     | [-1, 1] | Rudder left/right    |
| 3     | Thrust  | [0, 1]  | Engine throttle      |
| 4     | Gun     | [0, 1]  | Fire machine gun when > 0.5 |

## Reward Function

The reward is computed per step from plane state, health, and missile state:

| Condition                                | Reward                    | Purpose                                 |
|------------------------------------------|---------------------------|-----------------------------------------|
| Each timestep alive                      | +1.0                      | Incentivize survival                    |
| Missile evaded (all missiles gone)       | +100.0                    | Big bonus for confirmed evasion         |
| Hit by missile (health drops)            | -100.0                    | Strong penalty for failure              |
| Crashed (altitude < 50m or crashed flag) | -100.0                    | Don't fly into the ground               |
| Extreme manoeuvres                       | -0.05 * (p^2 + r^2 + y^2) | Discourage random flailing              |
| Control jerk                             | -0.02 * L2(delta action)^2| Discourage twitchy, unrealistic inputs  |
| Closest missile distance delta           | +/- 0.001 * delta metres  | Reward opening separation               |
| Missile approach speed                   | -0.0004 * m/s             | Penalize letting threats close quickly  |
| Close-range missile danger               | Up to -1.0                | Stronger penalty inside 1 km            |
| Low altitude (< 500m)                    | -5.0                      | Hard floor penalty                      |
| Low altitude (500-1000m)                 | -1.0                      | Soft floor penalty                      |
| High altitude (> 10,000m)                | -5.0                      | Hard ceiling penalty                    |
| High altitude (9,000-10,000m)            | -1.0                      | Soft ceiling penalty                    |
| Low-speed, high-pitch stall risk         | Up to -1.5                | Penalize nose-high low-energy flight    |
| Excessive G load (> 5g soft, > 9g hard)  | Quadratic penalty         | Discourage blackout-inducing maneuvers  |

## Flight-Envelope Telemetry

The simulator now exposes:

- **G load**: signed load factor computed from specific force in the aircraft
  frame and returned as `g_load`.
- **Angle of attack**: estimated from body-axis vertical/forward velocity and
  returned as `angle_of_attack`.

The env uses those values directly when available and only falls back to
deriving them from consecutive plane states if the simulator does not provide
them. `demo.py` prints the episode peak G load at the end, and the sandbox HUD
shows both G and AoA during flight.

Missile episodes also extend their timeout budget automatically from the live
missiles' reported `life_delay` and `life_time`, so a launched missile gets
enough sim time to burn out or self-destruct before the env can time out.

## Episode Setup

Each episode configures the scenario:

1. **Agent (F16):** Spawns at ~4000m altitude with slight random offset,
   flying at 300 m/s, thrust at 100%, gear retracted (forced every step).

2. **Enemy (Eurofighter):** Spawns 2km behind the agent at the same altitude,
   flying at 300 m/s. Controls set to fly straight (no maneuvering).

3. **SAM launcher:** Ground-based missile launcher, also targeted at agent.

4. **Targeting:** Enemy's targeting device is locked onto the agent via
   `set_target_id`. SAM launcher also targeted.

5. **Missiles:** Both enemy and SAM are rearmed with full loadout. After 30
   steps (~1 second of sim time), both fire from the first available slot.
   Missile guidance targets are explicitly set to the agent.

## Termination Conditions

| Condition      | Type       | When                                          |
|----------------|------------|-----------------------------------------------|
| Health dropped | terminated | `health_level < 0.99`                         |
| Crashed        | terminated | `crashed` flag or altitude < 50m              |
| Missile evaded | terminated | All tracked missiles destroyed or deactivated |
| Max steps      | truncated  | Timeout budget reached while missiles may still be active |

## Evasion Detection

Evasion is detected by querying actual missile state from the sandbox each
step. After firing, the env discovers the new missile IDs by diffing
`get_missiles_list` against a pre-fire snapshot. Each step, it queries
`get_missile_state` for each tracked missile.

The episode ends with "evaded" only when **every tracked missile** reports
`active: false` or `destroyed: true` — meaning it has actually
self-destructed, hit the ground, or run out of fuel. No time-based
shortcuts.

## Sim Timestep

The sandbox's physics timestep is set to 1/30s (double the default 1/60s).
This means each `step()` call covers more simulated time, reducing the
total number of steps needed for the same scenario duration and improving
training throughput.

## Network Efficiency

Each `step()` call uses our custom `STEP` (or `STEP_N` with action repeat)
server command: one TCP round-trip that applies controls, advances physics,
and returns the new plane state. Missile state queries add one round-trip
per tracked missile per step.
