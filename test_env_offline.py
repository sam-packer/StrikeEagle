"""Offline smoke test: verify the env class instantiates and spaces are correct.

Does NOT require the Dogfight Sandbox to be running.
"""

from src.missile_evasion_env import MissileEvasionEnv, OBS_DIM
import numpy as np


def main():
    env = MissileEvasionEnv(renderless=True)

    print(f"Observation space: {env.observation_space}")
    print(f"  shape: {env.observation_space.shape}")
    print(f"  expected dim: {OBS_DIM}")
    assert env.observation_space.shape == (OBS_DIM,), "Obs space shape mismatch"

    print(f"Action space: {env.action_space}")
    print(f"  low:  {env.action_space.low}")
    print(f"  high: {env.action_space.high}")
    assert env.action_space.shape == (5,), "Action space shape mismatch"

    # Sample random actions
    for _ in range(5):
        a = env.action_space.sample()
        assert a.shape == (5,)
        assert np.all(a >= env.action_space.low)
        assert np.all(a <= env.action_space.high)

    print("\nAll offline checks passed.")
    print(
        "\nTo run with the sandbox:\n"
        "  1. Start Dogfight Sandbox in Network mode\n"
        "  2. uv run python train.py --host localhost --port 50888\n"
        "  3. uv run python demo.py --model checkpoints/missile_evasion_sac_final.zip --host localhost --port 50888"
    )


if __name__ == "__main__":
    main()
