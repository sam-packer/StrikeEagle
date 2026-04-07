"""Run a trained missile-evasion agent with full 3D rendering in the sandbox."""

import argparse
import time

from stable_baselines3 import PPO, SAC

from src import dogfight_client as df
from src.missile_evasion_env import MissileEvasionEnv
from src.net_utils import get_default_host


def main():
    parser = argparse.ArgumentParser(description="Demo: missile evasion agent")
    parser.add_argument("--model", required=True, help="Path to trained model .zip")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--host", default=None, help="Sandbox host (auto-detects LAN IP if omitted)")
    parser.add_argument("--port", type=int, default=50888)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()

    env = MissileEvasionEnv(
        host=args.host or get_default_host(),
        port=args.port,
        renderless=False,  # Full 3D rendering for the demo
        max_steps=args.max_steps,
    )

    AlgoClass = PPO if args.algo == "ppo" else SAC
    model = AlgoClass.load(args.model)
    print(f"Loaded {args.algo.upper()} model from {args.model}")

    for ep in range(args.episodes):
        obs, _ = env.reset()

        # Set camera to chase the F16 with SFX
        df.set_camera_track(env.ally_id)
        df.set_track_view("back")

        total_reward = 0.0
        step = 0
        done = False

        print(f"\nEpisode {ep + 1}/{args.episodes} — missile fires at step {env.missile_fire_delay}...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            # Sandbox renders at its own native framerate; small sleep
            # just prevents flooding it with commands
            time.sleep(0.02)

        outcome = info.get("outcome", "unknown")
        print(
            f"Episode {ep + 1}/{args.episodes}: "
            f"steps={step}, reward={total_reward:.1f}, outcome={outcome}"
        )

        # Keep the sim ticking so the explosion/crash animation plays out
        # before resetting. In client_update_mode the sandbox only advances
        # when we call update_scene().
        post_frames = int(3.0 / (1 / 60))  # ~3 seconds at rendered 60Hz
        for _ in range(post_frames):
            df.update_scene()
            time.sleep(1 / 60)

    env.close()


if __name__ == "__main__":
    main()
