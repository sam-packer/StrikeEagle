"""Diagnostic script: connect to the Dogfight Sandbox and verify everything works.

Run this AFTER starting the sandbox in Network mode to confirm:
  - Connection works
  - Aircraft IDs are correct
  - State queries return sensible data
  - Control inputs are accepted
  - Missiles can be listed
  - Random actions don't crash the env

Usage:
    uv run diagnose
"""

import argparse
import math
import time

import numpy as np

from src import dogfight_client as df
from src.missile_evasion_env import MissileEvasionEnv
from src.net_utils import get_lan_ip


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose sandbox connection")
    parser.add_argument("--host", default=None, help="Sandbox host (auto-detects LAN IP if omitted)")
    parser.add_argument("--port", type=int, default=50888)
    parser.add_argument("--random-episodes", type=int, default=2,
                        help="Number of random-action episodes to run (0 to skip)")
    args = parser.parse_args()

    host = args.host or get_lan_ip()

    # ---- Phase 1: Raw connection test ----
    section("Phase 1: Connecting to Dogfight Sandbox")
    print(f"Connecting to {host}:{args.port} ...")
    df.connect(host, args.port)
    time.sleep(2)

    planes = df.get_planes_list()
    print(f"Connected! Planes: {planes}")
    if not planes:
        print("ERROR: No planes found. Is the sandbox in Network mode?")
        df.disconnect()
        return

    df.disable_log()
    df.set_client_update_mode(True)
    # Run in rendered mode so you can watch the planes in the 3D window.
    # Our patched update_scene synchronization makes this safe.
    print("Client-update mode enabled (rendered).")

    # ---- Phase 2: Query state of each aircraft ----
    section("Phase 2: Aircraft State")
    for plane_id in planes:
        state = df.get_plane_state(plane_id)
        pos = state.get("position", [0, 0, 0])
        alt = state.get("altitude", 0)
        speed = state.get("linear_speed", 0)
        health = state.get("health_level", 0)
        nationality = state.get("nationality", "unknown")
        print(
            f"  {plane_id:15s} | nat={str(nationality):10s} | "
            f"pos=({pos[0]:8.1f}, {pos[1]:8.1f}, {pos[2]:8.1f}) | "
            f"alt={alt:7.1f}m | speed={speed:6.1f}m/s | health={health:.2f}"
        )

    # ---- Phase 3: Check missiles ----
    section("Phase 3: Missile Systems")
    for plane_id in planes:
        slots = df.get_missiles_device_slots_state(plane_id)
        n_available = sum(1 for s in slots.get("missiles_slots", []) if s)
        print(f"  {plane_id:15s} | {n_available} missiles ready")

    # ---- Phase 4: Control Test (matching debug_connection.py pattern exactly) ----
    section("Phase 4: Control Test")
    ally_id = None
    enemy_id = None
    for p in planes:
        if "ally" in p and ally_id is None:
            ally_id = p
        if "ennemy" in p and enemy_id is None:
            enemy_id = p

    print(f"  ally={ally_id}, enemy={enemy_id}")

    # Reset ally
    df.reset_machine(ally_id)
    df.reset_machine_matrix(ally_id, 0, 4000, 0, 0, 0, 0)
    df.set_plane_thrust(ally_id, 1.0)
    df.set_plane_linear_speed(ally_id, 300)
    df.retract_gear(ally_id)
    df.get_plane_state(ally_id)  # sync
    df.update_scene()

    # Reset enemy
    df.reset_machine(enemy_id)
    df.reset_machine_matrix(enemy_id, 0, 4200, -5000, 0, 0, 0)
    df.set_plane_thrust(enemy_id, 0.8)
    df.set_plane_linear_speed(enemy_id, 250)
    df.retract_gear(enemy_id)
    df.get_plane_state(ally_id)  # sync
    df.update_scene()

    # Fly for 20 steps
    for i in range(20):
        df.set_plane_pitch(ally_id, 0.0)
        df.set_plane_roll(ally_id, 0.1)
        df.set_plane_yaw(ally_id, 0.0)
        df.get_plane_state(ally_id)  # sync
        df.update_scene()

    state_after = df.get_plane_state(ally_id)
    print(f"  After 20 steps: alt={state_after['altitude']:.1f}m, "
          f"speed={state_after['linear_speed']:.1f}m/s")
    print("  Controls working!")

    # ---- Phase 5: Missile Attack Demo ----
    section("Phase 5: Missile Attack Demo (watch the 3D window!)")

    # Reset both aircraft into an attack scenario
    df.reset_machine(ally_id)
    df.reset_machine(enemy_id)
    df.reset_machine_matrix(ally_id, 0, 4000, 0, 0, 0, 0)
    df.set_plane_thrust(ally_id, 1.0)
    df.set_plane_linear_speed(ally_id, 300)
    df.retract_gear(ally_id)

    # Enemy behind at same altitude, closing in
    df.reset_machine_matrix(enemy_id, 0, 4000, -2000, 0, 0, 0)
    df.set_plane_thrust(enemy_id, 1.0)
    df.set_plane_linear_speed(enemy_id, 300)
    df.retract_gear(enemy_id)
    df.set_plane_pitch(enemy_id, 0)
    df.set_plane_roll(enemy_id, 0)
    df.set_plane_yaw(enemy_id, 0)
    df.set_target_id(enemy_id, ally_id)
    df.set_health(ally_id, 1.0)
    df.rearm_machine(enemy_id)

    df.get_plane_state(ally_id)
    df.update_scene()

    # Set camera AFTER scenario is configured so SFX attaches to positioned aircraft
    df.set_camera_track(ally_id)
    df.set_track_view("back")

    # Run a few frames so camera settles on the aircraft
    for _ in range(5):
        df.step(ally_id, 0.0, 0.0, 0.0, 1.0)
        time.sleep(0.05)

    print("  Scenario set: F16 at 4000m, enemy 2km behind and closing...")
    print("  Flying for 25 steps to let targeting lock, then enemy fires missile...")

    # Fly straight for 25 steps to let enemy targeting system lock
    for i in range(25):
        df.step(ally_id, 0.0, 0.0, 0.0, 1.0)
        time.sleep(0.03)

    # Fire!
    enemy_missiles = df.get_machine_missiles_list(enemy_id)
    slots = df.get_missiles_device_slots_state(enemy_id)
    missile_slots = slots.get("missiles_slots", [])
    fired = False
    missile_name = None
    for slot_idx, available in enumerate(missile_slots):
        if available:
            missile_name = enemy_missiles[slot_idx] if slot_idx < len(enemy_missiles) else None
            # Force target lock on ally before firing so missile gets guidance
            df.fire_missile(enemy_id, slot_idx, target_id=ally_id)
            fired = True
            print(f"  MISSILE AWAY! {missile_name} fired from slot {slot_idx}, locked onto {ally_id}")
            break
    if not fired:
        print("  WARNING: No missiles available to fire!")

    # Now fly evasive maneuvers with the missile chasing
    print("  Evading! Watch the 3D window...")
    initial_health = 1.0
    hit = False
    for i in range(300):
        # Simple evasive pattern: alternating hard turns + climbs
        t = i / 30.0
        pitch = -0.5 + 0.3 * math.sin(t * 2)
        roll = 0.8 * math.sin(t * 3)
        state = df.step(ally_id, pitch, roll, 0.2, 1.0)
        time.sleep(0.03)  # ~30fps visual

        health = state.get("health_level", 1.0)
        alt = state.get("altitude", 0)

        if health < initial_health - 0.01:
            print(f"  HIT! Health dropped to {health:.2f} at step {i}, alt={alt:.0f}m")
            hit = True
            break

        if state.get("crashed") or alt < 100:
            print(f"  CRASHED at step {i}, alt={alt:.0f}m")
            hit = True
            break

        if i % 50 == 0 and i > 0:
            print(f"    step {i}: still alive, alt={alt:.0f}m, speed={state.get('linear_speed', 0):.0f}m/s")

    if not hit:
        print(f"  SURVIVED! Evaded the missile after 300 steps")

    # ---- Phase 6: Random-action episodes through the Gymnasium env ----
    if args.random_episodes > 0:
        section(f"Phase 6: Random-Action Episodes ({args.random_episodes})")

        if ally_id is None or enemy_id is None:
            print("ERROR: Could not find ally/enemy planes.")
            return

        env = MissileEvasionEnv(
            host=host,
            port=args.port,
            renderless=False,
            ally_id=ally_id,
            enemy_id=enemy_id,
            max_steps=300,
        )
        env._connected = True  # reuse existing connection

        for ep in range(args.random_episodes):
            obs, _ = env.reset()
            print(f"\n  Episode {ep + 1}: reset OK, obs shape={obs.shape}")
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

                if steps % 50 == 0:
                    print(f"    step {steps}: reward_so_far={total_reward:.1f}, "
                          f"obs[altitude]={obs[10]:.3f}")

            outcome = info.get("outcome", "unknown")
            print(f"  Episode {ep + 1} done: steps={steps}, "
                  f"reward={total_reward:.1f}, outcome={outcome}")

        env.close()

    section("ALL DIAGNOSTICS PASSED")
    print("Your setup is ready for training!\n")
    print("Next steps:")
    print("  uv run train --algo sac")


if __name__ == "__main__":
    main()
