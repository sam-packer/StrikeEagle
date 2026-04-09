"""Gymnasium environment for missile evasion in HarFang3D Dogfight Sandbox.

The agent controls a fighter jet and must evade incoming missiles fired by an
enemy aircraft.  The enemy is placed nearby, given a target lock on the agent,
and commanded to fire a missile at episode start.  The agent's job is to
survive until the missiles are actually gone (destroyed/deactivated).
"""

import math
import random
import time

import gymnasium as gym
import numpy as np

from src import dogfight_client as df
from src.net_utils import get_default_host

# ---------------------------------------------------------------------------
# Normalisation constants (matching HIRL4UCAV conventions)
# ---------------------------------------------------------------------------
NORM_POS = 10_000.0  # metres
NORM_EULER = math.pi  # radians
NORM_SPEED = 800.0  # m/s
NORM_MISSILE_POS = 20_000.0
NORM_MISSILE_SPEED = 2_000.0
GRAVITY_MPS2 = 9.80665
GRAVITY_VECTOR = np.array([0.0, -GRAVITY_MPS2, 0.0], dtype=np.float32)

# Altitude safety band (metres)
ALT_MIN = 500.0
ALT_MAX = 10_000.0

# Maximum number of missiles we track in the observation (AAM + SAM)
MAX_MISSILES = 2

# Per-missile observation: rel_pos(3) + rel_vel(3) + distance(1) + closing_rate(1)
MISSILE_OBS_DIM = 8

# Agent obs: pos(3) + vel(3) + euler(3) + speed(1) + altitude(1) = 11
AGENT_OBS_DIM = 11

OBS_DIM = AGENT_OBS_DIM + MAX_MISSILES * MISSILE_OBS_DIM  # 27

# Number of sim sub-steps per env.step() when rendering is on.
# In renderless mode we use 1 because each update_scene call is one physics
# tick and we want maximum throughput.
RENDER_SUBSTEPS = 8
RENDERLESS_SUBSTEPS = 1

# Minimum steps after missile fire before we start checking missile state,
# giving the sim time to register the missile objects.
MISSILE_SETTLE_STEPS = 5

# Reward shaping constants.
MISSILE_DISTANCE_DELTA_SCALE = 0.001
MISSILE_CLOSING_PENALTY_SCALE = 0.0004
MISSILE_DANGER_DISTANCE = 1_000.0
MISSILE_TIMEOUT_BUFFER_SECONDS = 3.0
G_SOFT_LIMIT = 5.0
G_HARD_LIMIT = 9.0
STALL_SPEED_SOFT = 150.0
STALL_SPEED_HARD = 100.0
STALL_PITCH_SOFT = 15.0
STALL_PITCH_HARD = 25.0
ACTION_SMOOTHNESS_SCALE = 0.02
GUN_FIRE_THRESHOLD = 0.5


class MissileEvasionEnv(gym.Env):
    """Gymnasium env: evade a missile fired by an enemy aircraft.

    The observation is a flat float32 vector (see OBS_DIM).
    The action is [pitch, roll, yaw, thrust, gun].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        host: str | None = None,
        port: int = 50888,
        renderless: bool = True,
        ally_id: str = "ally_1",
        enemy_id: str = "ennemy_1",
        sam_id: str = "Ennemy_Missile_launcher_1",
        max_steps: int = 1500,
        missile_fire_delay: int = 30,
    ):
        super().__init__()

        self.host = host or get_default_host()
        self.port = port
        self.renderless = renderless
        self.ally_id = ally_id
        self.enemy_id = enemy_id
        self.sam_id = sam_id
        self.max_steps = max_steps
        self.missile_fire_delay = missile_fire_delay

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._connected = False
        self._step_count = 0
        self._missile_launched = False
        self._missile_fire_step = 0
        self._tracked_missile_ids: list[str] = []
        self._initial_health = 1.0
        self._gear_retracted = False
        self._substeps = RENDERLESS_SUBSTEPS if renderless else RENDER_SUBSTEPS
        # Snapshot of missile IDs before firing, so we can diff to find new ones
        self._pre_fire_missile_ids: set[str] = set()
        self._prev_plane_state: dict | None = None
        self._prev_action: np.ndarray | None = None
        self._prev_closest_missile_distance: float | None = None
        self._max_g_load = 0.0
        self._effective_max_steps = max_steps

    def _normalize_action(self, action) -> np.ndarray:
        """Normalize actions to [pitch, roll, yaw, thrust, gun]."""
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 5:
            raise ValueError(f"Expected action with 5 elements, got shape {arr.shape}")
        return arr

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def connect(self):
        """Connect to the Dogfight Sandbox server (call once before reset)."""
        if self._connected:
            return
        df.connect(self.host, self.port)
        # Official examples wait 2s after connect for the sandbox to settle
        time.sleep(2)
        df.disable_log()
        if self.renderless:
            df.set_client_update_mode(True)
            # 2x sim timestep: each step covers 1/30s instead of 1/60s,
            # halving the number of steps needed for the same scenario duration
            df.set_timestep(1 / 30)
            df.set_renderless_mode(True)
            # Wait for sandbox to finish transitioning to renderless mode
            while not df.get_running().get("running", False):
                pass
        self._connected = True

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.connect()

        self._step_count = 0
        self._missile_launched = False
        self._missile_fire_step = 0
        self._tracked_missile_ids = []
        self._gear_retracted = False
        self._prev_plane_state = None
        self._prev_action = None
        self._prev_closest_missile_distance = None
        self._max_g_load = 0.0
        self._effective_max_steps = self.max_steps

        # Reset both aircraft
        df.reset_machine(self.ally_id)
        df.reset_machine(self.enemy_id)

        # Place agent: altitude ~4000 m, flying forward
        ax = random.randint(-500, 500)
        az = random.randint(-500, 500)
        df.reset_machine_matrix(self.ally_id, ax, 4000, az, 0, 0, 0)
        df.set_plane_thrust(self.ally_id, 1.0)
        df.set_plane_linear_speed(self.ally_id, 300)
        df.retract_gear(self.ally_id)
        df.deactivate_machine_gun(self.ally_id)

        # Place enemy: ~2 km behind the agent, same altitude, chasing
        df.reset_machine_matrix(
            self.enemy_id, ax, 4000, az - 2000, 0, 0, 0
        )
        df.set_plane_thrust(self.enemy_id, 1.0)
        df.set_plane_linear_speed(self.enemy_id, 300)
        df.retract_gear(self.enemy_id)
        # Set enemy controls to fly straight — stays set for the whole episode
        df.set_plane_pitch(self.enemy_id, 0)
        df.set_plane_roll(self.enemy_id, 0)
        df.set_plane_yaw(self.enemy_id, 0)

        # Sync + advance one frame so positions take effect
        df.get_plane_state(self.ally_id)
        df.update_scene()

        # Enemy targets our agent
        df.set_target_id(self.enemy_id, self.ally_id)

        # Restore health and rearm enemy missiles + SAM
        df.set_health(self.ally_id, 1.0)
        df.set_health(self.enemy_id, 1.0)
        df.rearm_machine(self.enemy_id)
        df.set_target_id(self.sam_id, self.ally_id)
        df.rearm_machine(self.sam_id)

        # Sync + advance so target/health/rearm take effect
        df.get_plane_state(self.ally_id)
        df.update_scene()

        # Snapshot existing missile IDs before firing so we can diff later
        self._pre_fire_missile_ids = set(df.get_missiles_list())

        self._initial_health = 1.0

        plane_state = df.get_plane_state(self.ally_id)
        self._prev_plane_state = plane_state
        obs = self._build_obs(plane_state, [])
        return obs, {}

    def step(self, action, n=1):
        action = self._normalize_action(action)
        self._step_count += n

        # Fire missile before advancing (so missile is in flight this frame)
        if not self._missile_launched and self._step_count >= self.missile_fire_delay:
            self._fire_enemy_missile()

        # Retract gear once after the first physics frame has settled
        if not self._gear_retracted:
            df.retract_gear(self.ally_id)
            self._gear_retracted = True

        if float(action[4]) >= GUN_FIRE_THRESHOLD:
            df.activate_machine_gun(self.ally_id)
        else:
            df.deactivate_machine_gun(self.ally_id)

        # Single round-trip: apply controls + advance sim n frames + return state
        if n > 1:
            plane_state = df.step_n(
                self.ally_id,
                float(action[0]), float(action[1]),
                float(action[2]), float(action[3]),
                n,
            )
        else:
            plane_state = df.step(
                self.ally_id,
                float(action[0]), float(action[1]),
                float(action[2]), float(action[3]),
            )
        health = plane_state.get("health_level", 1.0)

        # Query missile states
        missile_states = self._get_missile_states()

        # Discover newly fired missiles if we haven't found them yet
        if self._missile_launched and not self._tracked_missile_ids:
            steps_since_fire = self._step_count - self._missile_fire_step
            if steps_since_fire >= MISSILE_SETTLE_STEPS:
                current_ids = set(df.get_missiles_list())
                new_ids = current_ids - self._pre_fire_missile_ids
                self._tracked_missile_ids = list(new_ids)
                if self._tracked_missile_ids:
                    missile_states = self._get_missile_states()

        obs = self._build_obs(plane_state, missile_states)
        reward, terminated, truncated, info = self._compute_reward_and_done(
            action, plane_state, health, missile_states, n
        )
        self._prev_plane_state = plane_state
        self._prev_action = np.array(action, dtype=np.float32, copy=True)

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._connected:
            if self.renderless:
                df.set_client_update_mode(False)
                df.set_renderless_mode(False)
            df.disconnect()
            self._connected = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fire_enemy_missile(self):
        """Fire both AAM (from enemy aircraft) and SAM (from ground launcher)."""
        # Fire AAM from enemy aircraft
        slots = df.get_missiles_device_slots_state(self.enemy_id)
        missile_slots = slots.get("missiles_slots", [])

        for slot_idx, available in enumerate(missile_slots):
            if available:
                df.fire_missile(self.enemy_id, slot_idx, target_id=self.ally_id)
                break

        # Fire SAM from ground launcher
        sam_slots = df.get_missiles_device_slots_state(self.sam_id)
        sam_missile_slots = sam_slots.get("missiles_slots", [])

        for slot_idx, available in enumerate(sam_missile_slots):
            if available:
                df.fire_missile(self.sam_id, slot_idx, target_id=self.ally_id)
                break

        self._missile_launched = True
        self._missile_fire_step = self._step_count

    def _get_missile_states(self) -> list[dict]:
        """Query state for all tracked missiles. Returns list of state dicts."""
        states = []
        for missile_id in self._tracked_missile_ids:
            state = df.get_missile_state(missile_id)
            state["missile_id"] = missile_id
            states.append(state)
        return states

    def _all_missiles_gone(self, missile_states: list[dict]) -> bool:
        """True when every tracked missile is destroyed or deactivated."""
        if not self._tracked_missile_ids:
            return False  # Haven't discovered them yet
        for ms in missile_states:
            if ms.get("active", False) and not ms.get("destroyed", False):
                return False
        return True

    def _get_obs(self):
        """Query state and build obs — used only by reset()."""
        plane_state = df.get_plane_state(self.ally_id)
        return self._build_obs(plane_state, [])

    def _build_obs(self, plane, missile_states: list[dict]):
        """Build the flat observation vector from cached state dicts."""
        pos = np.array(plane["position"], dtype=np.float32) / NORM_POS
        vel = np.array(plane["move_vector"], dtype=np.float32) / NORM_SPEED
        euler = np.array(plane["Euler_angles"], dtype=np.float32) / NORM_EULER
        speed = np.float32(plane.get("linear_speed", 0.0) / NORM_SPEED)
        alt = np.float32(plane.get("altitude", 0.0) / NORM_POS)

        agent_obs = np.concatenate([pos, vel, euler, [speed], [alt]])

        # Missile observations — fill slots for up to MAX_MISSILES
        missile_obs = np.zeros(MAX_MISSILES * MISSILE_OBS_DIM, dtype=np.float32)
        a_pos = np.array(plane["position"], dtype=np.float32)
        a_vel = np.array(plane["move_vector"], dtype=np.float32)

        for i, ms in enumerate(missile_states[:MAX_MISSILES]):
            if not ms.get("active", False) or ms.get("destroyed", False):
                continue
            if "position" not in ms:
                continue

            m_pos = np.array(ms["position"], dtype=np.float32)
            m_vel = np.array(ms["move_vector"], dtype=np.float32)

            rel_pos = (m_pos - a_pos) / NORM_MISSILE_POS
            rel_vel = (m_vel - a_vel) / NORM_MISSILE_SPEED
            distance = np.float32(
                np.linalg.norm(m_pos - a_pos) / NORM_MISSILE_POS
            )
            diff = m_pos - a_pos
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-6:
                closing_rate = np.float32(
                    np.dot(m_vel - a_vel, diff / diff_norm) / NORM_MISSILE_SPEED
                )
            else:
                closing_rate = np.float32(0.0)

            offset = i * MISSILE_OBS_DIM
            missile_obs[offset:offset + MISSILE_OBS_DIM] = np.concatenate(
                [rel_pos, rel_vel, [distance], [closing_rate]]
            )

        return np.concatenate([agent_obs, missile_obs])

    def _get_active_missile_metrics(self, plane, missile_states: list[dict]) -> list[dict]:
        """Return distance/closure metrics for active missiles."""
        metrics = []
        a_pos = np.array(plane["position"], dtype=np.float32)
        a_vel = np.array(plane["move_vector"], dtype=np.float32)

        for ms in missile_states:
            if not ms.get("active", False) or ms.get("destroyed", False):
                continue
            if "position" not in ms:
                continue

            m_pos = np.array(ms["position"], dtype=np.float32)
            m_vel = np.array(ms["move_vector"], dtype=np.float32)
            rel_pos = m_pos - a_pos
            distance = float(np.linalg.norm(rel_pos))
            if distance > 1e-6:
                rel_dir = rel_pos / distance
                approach_speed = float(-np.dot(m_vel - a_vel, rel_dir))
            else:
                approach_speed = 0.0

            metrics.append(
                {
                    "missile_id": ms.get("missile_id"),
                    "distance": distance,
                    "approach_speed": max(0.0, approach_speed),
                }
            )

        return metrics

    def _compute_flight_metrics(self, plane, n: int) -> dict:
        """Estimate flight-envelope metrics from consecutive plane states."""
        g_load = plane.get("g_load")
        if g_load is not None:
            return {
                "g_load": float(g_load),
                "angle_of_attack": abs(float(plane.get("angle_of_attack", 0.0))),
                "pitch_attitude": abs(float(plane.get("pitch_attitude", 0.0))),
                "vertical_speed": float(plane.get("vertical_speed", 0.0)),
            }

        move_vector = np.array(plane["move_vector"], dtype=np.float32)
        timestep = max(float(plane.get("timestep", 1 / 30)) * max(n, 1), 1e-6)

        if self._prev_plane_state is None:
            world_accel = np.zeros(3, dtype=np.float32)
        else:
            prev_move = np.array(
                self._prev_plane_state.get("move_vector", move_vector),
                dtype=np.float32,
            )
            world_accel = (move_vector - prev_move) / timestep

        # Proper acceleration is the non-gravitational acceleration the pilot feels.
        specific_force = world_accel - GRAVITY_VECTOR
        g_load = float(np.linalg.norm(specific_force) / GRAVITY_MPS2)

        return {
            "g_load": g_load,
            "angle_of_attack": abs(float(plane.get("angle_of_attack", 0.0))),
            "pitch_attitude": abs(float(plane.get("pitch_attitude", 0.0))),
            "vertical_speed": float(plane.get("vertical_speed", 0.0)),
        }

    def _count_active_missiles(self, missile_states: list[dict]) -> int:
        return sum(
            1
            for ms in missile_states
            if ms.get("active", False) and not ms.get("destroyed", False)
        )

    def _update_timeout_budget(self, plane: dict, missile_states: list[dict]) -> int:
        """Extend the timeout so launched missiles can actually burn out."""
        timeout_budget = self.max_steps
        default_timestep = max(float(plane.get("timestep", 1 / 30)), 1e-6)

        for ms in missile_states:
            if not ms.get("active", False) or ms.get("destroyed", False):
                continue

            life_delay = ms.get("life_delay")
            life_time = ms.get("life_time")
            if life_delay is None or life_time is None:
                continue

            missile_timestep = max(float(ms.get("timestep", default_timestep)), 1e-6)
            remaining_life = max(0.0, float(life_delay) - float(life_time))
            timeout_budget = max(
                timeout_budget,
                self._step_count
                + int(math.ceil((remaining_life + MISSILE_TIMEOUT_BUFFER_SECONDS) / missile_timestep)),
            )

        self._effective_max_steps = max(self._effective_max_steps, timeout_budget)
        return self._effective_max_steps

    def _compute_reward_and_done(self, action, plane, health, missile_states, n):
        info = {}
        reward = 0.0
        terminated = False
        truncated = False

        alt = plane.get("altitude", 4000.0)
        speed = plane.get("linear_speed", 300.0)
        action = np.asarray(action, dtype=np.float32)
        flight_metrics = self._compute_flight_metrics(plane, n)
        g_load = flight_metrics["g_load"]
        angle_of_attack = flight_metrics["angle_of_attack"]
        pitch_attitude = flight_metrics["pitch_attitude"]
        self._max_g_load = max(self._max_g_load, g_load)
        active_missiles_remaining = self._count_active_missiles(missile_states)
        timeout_budget = self._update_timeout_budget(plane, missile_states)

        info["g_load"] = g_load
        info["max_g_load"] = self._max_g_load
        info["angle_of_attack"] = angle_of_attack
        info["pitch_attitude"] = pitch_attitude
        info["vertical_speed"] = flight_metrics["vertical_speed"]
        info["tracked_missiles"] = len(self._tracked_missile_ids)
        info["active_missiles_remaining"] = active_missiles_remaining
        info["timeout_step_budget"] = timeout_budget
        info["gun_active"] = float(action[4]) >= GUN_FIRE_THRESHOLD

        # +1 per step alive (survival reward)
        reward += 1.0

        # Penalty for extreme manoeuvres (encourages smooth evasion)
        action_magnitude = float(np.sum(np.square(action[:3])))
        reward -= 0.05 * action_magnitude
        if self._prev_action is not None:
            action_delta = action - self._prev_action
            reward -= ACTION_SMOOTHNESS_SCALE * float(np.sum(np.square(action_delta[:3])))
            info["control_delta"] = float(np.linalg.norm(action_delta[:3]))

        # Altitude penalty — stay in safe band
        if alt < ALT_MIN:
            reward -= 5.0
        elif alt < 1000:
            reward -= 1.0
        if alt > ALT_MAX:
            reward -= 5.0
        elif alt > 9000:
            reward -= 1.0

        # Speed penalty — don't stall. Below 150 m/s the aircraft is
        # dangerously slow; below 80 m/s it's effectively stalled.
        if speed < 80:
            reward -= 5.0
        elif speed < 150:
            reward -= 1.0

        # Low-energy, nose-high flight is the closest thing this simulator has
        # to an explicit stall regime, so penalize it directly.
        stall_risk = 0.0
        stall_angle = angle_of_attack if angle_of_attack > 0 else pitch_attitude
        if speed < STALL_SPEED_SOFT and stall_angle > STALL_PITCH_SOFT:
            speed_risk = min(
                1.0,
                max(0.0, (STALL_SPEED_SOFT - speed) / (STALL_SPEED_SOFT - STALL_SPEED_HARD)),
            )
            pitch_risk = min(
                1.0,
                max(0.0, (stall_angle - STALL_PITCH_SOFT) / (STALL_PITCH_HARD - STALL_PITCH_SOFT)),
            )
            stall_risk = speed_risk * pitch_risk
            reward -= 1.5 * stall_risk
        info["stall_risk"] = stall_risk

        # Penalize blackout-inducing maneuvers.
        g_over_soft = max(0.0, g_load - G_SOFT_LIMIT)
        if g_over_soft > 0:
            reward -= 0.25 * g_over_soft * g_over_soft
        if g_load > G_HARD_LIMIT:
            reward -= 2.0 + (g_load - G_HARD_LIMIT)

        missile_metrics = self._get_active_missile_metrics(plane, missile_states)
        if missile_metrics:
            closest_distance = min(m["distance"] for m in missile_metrics)
            max_approach_speed = max(m["approach_speed"] for m in missile_metrics)
            info["closest_missile_distance"] = closest_distance
            info["max_missile_approach_speed"] = max_approach_speed

            if self._prev_closest_missile_distance is not None:
                distance_delta = np.clip(
                    closest_distance - self._prev_closest_missile_distance,
                    -200.0,
                    200.0,
                )
                reward += MISSILE_DISTANCE_DELTA_SCALE * float(distance_delta)

            reward -= MISSILE_CLOSING_PENALTY_SCALE * max_approach_speed
            if closest_distance < MISSILE_DANGER_DISTANCE:
                reward -= (MISSILE_DANGER_DISTANCE - closest_distance) / MISSILE_DANGER_DISTANCE

            self._prev_closest_missile_distance = closest_distance
        else:
            self._prev_closest_missile_distance = None

        # Check if hit by missile (health dropped)
        if health < self._initial_health - 0.01:
            reward -= 100.0
            terminated = True
            info["outcome"] = "hit"

        # Check if crashed
        if plane.get("crashed", False) or plane.get("wreck", False):
            reward -= 100.0
            terminated = True
            info["outcome"] = "crashed"

        # Ground collision
        if alt < 50:
            reward -= 100.0
            terminated = True
            info["outcome"] = "ground_collision"

        # Evasion: only when all missiles are actually gone
        if (
            self._missile_launched
            and not terminated
            and self._all_missiles_gone(missile_states)
        ):
            reward += 100.0
            terminated = True
            info["outcome"] = "evaded"

        # Max steps reached
        if self._step_count >= timeout_budget and not terminated:
            if self._missile_launched and self._all_missiles_gone(missile_states):
                reward += 100.0
                terminated = True
                info["outcome"] = "evaded"
            else:
                truncated = True
                if self._missile_launched:
                    if active_missiles_remaining > 0:
                        info["outcome"] = "timeout_active_missiles"
                    else:
                        info["outcome"] = "timeout_unknown_missile_state"
                else:
                    info["outcome"] = "timeout_no_missile"

        return reward, terminated, truncated, info
