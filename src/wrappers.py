"""Gymnasium wrappers for the missile evasion environment."""

import gymnasium as gym


class ActionRepeat(gym.Wrapper):
    """Repeat each action for *n* sim frames in a single network call.

    Uses the sandbox's STEP_N command to advance *n* physics frames with
    the same controls in one round-trip, then computes obs/reward from
    the resulting state.  The env's internal step counter is advanced by
    *n* so episode length bookkeeping stays correct.

    NOTE: When using this wrapper, divide --total-timesteps by the repeat
    count to cover the same amount of sim time (e.g. 200k steps with
    repeat=1 ≈ 50k steps with repeat=4).
    """

    def __init__(self, env: gym.Env, repeat: int = 4):
        super().__init__(env)
        assert repeat >= 1, "repeat must be >= 1"
        self._repeat = repeat

    def step(self, action):
        return self.env.step(action, n=self._repeat)
