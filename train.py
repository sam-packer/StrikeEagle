"""Train a missile-evasion agent with Stable-Baselines3 PPO or SAC."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.missile_evasion_env import MissileEvasionEnv
from src.net_utils import get_default_host
from src.wrappers import ActionRepeat


def make_env(args, renderless=True):
    env = MissileEvasionEnv(
        host=args.host or get_default_host(),
        port=args.port,
        renderless=renderless,
        max_steps=args.max_steps,
    )
    if args.action_repeat > 1:
        env = ActionRepeat(env, repeat=args.action_repeat)
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser(description="Train missile evasion agent")
    parser.add_argument("--host", default=None, help="Sandbox host (auto-detects LAN IP if omitted)")
    parser.add_argument("--port", type=int, default=50888)
    parser.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to model zip to resume from")
    parser.add_argument(
        "--action-repeat", type=int, default=4,
        help="Repeat each action for N sim frames in one network call (default: 4). "
             "Divide --total-timesteps by this value to cover the same sim time "
             "(e.g. 200k/4 = 50k timesteps).",
    )
    parser.add_argument("--render", action="store_true", help="Train with 3D rendering (slow but visual)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    save_dir = Path(args.save_dir)
    log_dir.mkdir(exist_ok=True)
    save_dir.mkdir(exist_ok=True)

    env = make_env(args, renderless=not args.render)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(save_dir),
        name_prefix="missile_evasion",
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        AlgoClass = PPO if args.algo == "ppo" else SAC
        model = AlgoClass.load(args.resume, env=env, tensorboard_log=str(log_dir))
    else:
        if args.algo == "ppo":
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=str(log_dir),
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=dict(net_arch=[256, 256]),
            )
        else:
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=str(log_dir),
                learning_rate=3e-4,
                buffer_size=500_000,
                batch_size=1024,
                gamma=0.99,
                tau=0.005,
                ent_coef="auto",
                gradient_steps=4,
                policy_kwargs=dict(net_arch=[512, 512]),
            )

    print(f"Training {args.algo.upper()} for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb],
        tb_log_name=f"missile_evasion_{args.algo}",
        progress_bar=True,
    )

    final_path = save_dir / f"missile_evasion_{args.algo}_final"
    model.save(str(final_path))
    print(f"Final model saved to {final_path}")

    env.close()


if __name__ == "__main__":
    main()
