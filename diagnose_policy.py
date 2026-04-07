"""Diagnose what the trained policy is actually outputting.

Feeds representative observations through the model and prints the predicted
actions to check for degenerate behavior (zero thrust, constant pitch, etc.).
"""

import argparse
import numpy as np
from stable_baselines3 import SAC, PPO


OBS_DIM = 27
NORM_POS = 10_000.0
NORM_SPEED = 800.0
NORM_EULER = np.pi


def make_obs(alt=4000, speed=300, pitch_angle=0.0, missile_dist=None, missile_closing=None):
    """Build a synthetic observation vector."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    # Position: (0, alt, 0)
    obs[0] = 0.0 / NORM_POS
    obs[1] = alt / NORM_POS
    obs[2] = 0.0 / NORM_POS
    # Velocity: flying forward at given speed
    obs[3] = 0.0 / NORM_SPEED
    obs[4] = 0.0 / NORM_SPEED
    obs[5] = speed / NORM_SPEED
    # Euler: level flight with given pitch
    obs[6] = 0.0 / NORM_EULER  # roll
    obs[7] = pitch_angle / NORM_EULER  # pitch
    obs[8] = 0.0 / NORM_EULER  # yaw
    # Speed scalar + altitude
    obs[9] = speed / NORM_SPEED
    obs[10] = alt / NORM_POS

    # Missile 1 (if provided)
    if missile_dist is not None:
        obs[11] = 0.0  # rel_pos x
        obs[12] = 0.0  # rel_pos y
        obs[13] = -missile_dist / 20_000.0  # rel_pos z (behind)
        obs[14] = 0.0  # rel_vel x
        obs[15] = 0.0  # rel_vel y
        obs[16] = (missile_closing or 500) / 2_000.0  # rel_vel z (closing)
        obs[17] = missile_dist / 20_000.0  # distance
        obs[18] = (missile_closing or 500) / 2_000.0  # closing rate

    return obs


def describe_action(action):
    """Human-readable action description."""
    pitch, roll, yaw, thrust = action
    parts = []
    if abs(pitch) > 0.1:
        parts.append(f"pitch {'UP' if pitch > 0 else 'DOWN'} {abs(pitch):.2f}")
    if abs(roll) > 0.1:
        parts.append(f"roll {'RIGHT' if roll > 0 else 'LEFT'} {abs(roll):.2f}")
    if abs(yaw) > 0.1:
        parts.append(f"yaw {'RIGHT' if yaw > 0 else 'LEFT'} {abs(yaw):.2f}")
    parts.append(f"thrust {thrust:.2f}")
    return ", ".join(parts) if parts else "neutral"


def main():
    parser = argparse.ArgumentParser(description="Diagnose trained policy outputs")
    parser.add_argument("--model", required=True, help="Path to model .zip")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    args = parser.parse_args()

    AlgoClass = PPO if args.algo == "ppo" else SAC
    model = AlgoClass.load(args.model)
    print(f"Loaded {args.algo.upper()} from {args.model}")
    print(f"Observation space: {model.observation_space}")
    print(f"Action space: {model.action_space}")
    print()

    scenarios = [
        ("Level flight, 4000m, 300m/s, no missile", make_obs(alt=4000, speed=300)),
        ("Level flight, 4000m, 300m/s, missile 2km behind closing at 500m/s", make_obs(alt=4000, speed=300, missile_dist=2000, missile_closing=500)),
        ("Level flight, 4000m, 300m/s, missile 500m behind closing at 800m/s", make_obs(alt=4000, speed=300, missile_dist=500, missile_closing=800)),
        ("Low altitude 800m, 300m/s, no missile", make_obs(alt=800, speed=300)),
        ("High altitude 9500m, 300m/s, no missile", make_obs(alt=9500, speed=300)),
        ("Slow speed 100m/s, 4000m, no missile", make_obs(alt=4000, speed=100)),
        ("Fast speed 600m/s, 4000m, no missile", make_obs(alt=4000, speed=600)),
        ("Nose up 30deg, 4000m, 300m/s", make_obs(alt=4000, speed=300, pitch_angle=0.52)),
        ("Nose down 30deg, 4000m, 300m/s", make_obs(alt=4000, speed=300, pitch_angle=-0.52)),
    ]

    print("=" * 90)
    print(f"{'Scenario':<65} {'Action'}")
    print("=" * 90)

    for name, obs in scenarios:
        action, _ = model.predict(obs, deterministic=True)
        print(f"{name:<65} [{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}, {action[3]:.3f}]")
        print(f"{'':65} {describe_action(action)}")
        print()

    # Stats: run 100 random observations and show action distributions
    print("=" * 90)
    print("Action statistics over 100 random observations:")
    print("=" * 90)
    actions = []
    for _ in range(100):
        obs = make_obs(
            alt=np.random.uniform(500, 10000),
            speed=np.random.uniform(50, 700),
            pitch_angle=np.random.uniform(-1, 1),
            missile_dist=np.random.uniform(100, 5000) if np.random.random() > 0.3 else None,
            missile_closing=np.random.uniform(100, 1500) if np.random.random() > 0.3 else None,
        )
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)

    actions = np.array(actions)
    labels = ["Pitch", "Roll", "Yaw", "Thrust"]
    print(f"{'':10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    for i, label in enumerate(labels):
        print(f"{label:10} {actions[:, i].mean():+8.3f} {actions[:, i].std():8.3f} {actions[:, i].min():+8.3f} {actions[:, i].max():+8.3f}")


if __name__ == "__main__":
    main()
