# Strike Eagle - Training Guide

## Prerequisites

1. Python 3.13+ with [uv](https://docs.astral.sh/uv/)
2. NVIDIA GPU with CUDA support (tested on RTX 5090)
3. Dependencies installed: `uv sync`
4. Sandbox dependencies: `cd refs/dogfight-sandbox-hg2 && python -m pip install -r source/requirements.txt`

## Quick Start

```bash
# Terminal 1: Start the simulator
cd refs/dogfight-sandbox-hg2
1-start.bat
# Select "Network mode" (first mission, press LEFT then SPACE)

# Terminal 2: Verify connection
uv run diagnose

# Terminal 3: Train
uv run train --algo sac --total-timesteps 200000
```

## Training Commands

### Basic training
```bash
uv run train
```

### With options
```bash
uv run train --algo sac --total-timesteps 500000 --checkpoint-freq 10000
```

### Resume from checkpoint
```bash
uv run train --resume checkpoints/missile_evasion_50000_steps.zip
```

### Action repeat (frame skipping)

By default, the agent makes one decision every 4 sim frames instead of every
frame. This speeds up training by reducing both network round-trips and neural
network forward passes.

**Important:** When using action repeat, divide `--total-timesteps` by the
repeat count to cover the same amount of sim time. For example:

```bash
# These cover the same sim time:
uv run train --action-repeat 1 --total-timesteps 200000   # no repeat
uv run train --action-repeat 4 --total-timesteps 50000    # 4x repeat (default)
```

Use `--action-repeat 1` to disable frame skipping if you need per-frame
decisions.

### All options
| Flag | Default | Description |
|------|---------|-------------|
| `--algo` | `sac` | `sac` or `ppo` |
| `--total-timesteps` | `200000` | Total training steps |
| `--max-steps` | `1500` | Max steps per episode |
| `--action-repeat` | `4` | Frames per decision (divide timesteps by this) |
| `--checkpoint-freq` | `10000` | Save every N steps |
| `--log-dir` | `logs/` | TensorBoard directory |
| `--save-dir` | `checkpoints/` | Checkpoint directory |
| `--resume` | - | Path to model .zip |
| `--host` | auto | Sandbox host IP |
| `--port` | `50888` | Sandbox port |

## Monitoring

### TensorBoard
```bash
uv run tensorboard --logdir logs
# Open http://localhost:6006
```

Key metrics:
- **ep_rew_mean** — Average episode reward. Should trend upward.
- **ep_len_mean** — Average episode length. Longer = surviving longer.
- **actor_loss** — Policy network loss. Should decrease.
- **critic_loss** — Value network loss. Should stabilize.

### Interpreting Results

| ep_rew_mean | Interpretation |
|-------------|----------------|
| < 50 | Agent crashes or gets hit quickly |
| 50-200 | Basic survival, no evasion |
| 200-350 | Surviving most episodes, occasional evasion |
| 350+ | Consistent evasion |

## Demo / Evaluation

```bash
# With 3D rendering, camera tracking, and SFX
uv run demo --model checkpoints/missile_evasion_sac_final.zip

# More episodes
uv run demo --model checkpoints/missile_evasion_sac_final.zip --episodes 10
```

The demo runs in rendered mode with the camera following the F16. Use
Windows Game Bar (Win+G) or OBS to record video.

## Algorithm Choice

### SAC (Soft Actor-Critic) — Recommended
- Off-policy: more sample efficient
- Automatic entropy tuning: balances exploration/exploitation
- Best for continuous action spaces
- Handles the survival + evasion reward structure well

### PPO (Proximal Policy Optimization)
- On-policy: needs more samples but more stable
- Simpler to tune
- Good fallback if SAC doesn't converge
- Use with: `--algo ppo`

## Training Speed

The bottleneck is the physics simulation (~2-5ms per `main.update()`), not
the GPU or network. Expected throughput:

| Mode | Steps/sec | 200k time |
|------|-----------|-----------|
| Renderless (training) | ~200-400 | ~10-15 min |
| Rendered (demo) | ~30 | N/A |

GPU utilization will be ~30-40% because the simulation is CPU-bound. We
compensate with `gradient_steps=4` (4 gradient updates per env step) and
`batch_size=1024` to maximize learning per expensive simulation step.

## Tips

- **Always train in renderless mode** (the default). Rendered mode is ~10x
  slower and only needed for demos.
- **Start with 50k-200k steps** to verify the pipeline works, then scale up.
- **Check TensorBoard early** — if reward is flat after 50k steps, something
  is wrong with the scenario (missile not tracking, enemy too far, etc.).
- **The sandbox must stay running** throughout training. If it crashes,
  restart it and resume with `--resume`.
- **Ctrl+C training** saves no final model — use the latest checkpoint in
  `checkpoints/` instead.
