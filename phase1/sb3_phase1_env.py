from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from phase1.mec_phase1_env import MECPhase1MAPPOEnv


class SB3Phase1JointEnv(gym.Env):
    """
    Single-agent wrapper around the phase-1 multi-agent environment.

    The SB3 agent outputs a joint action covering all UEs and the server.
    Observations are concatenated into one flat vector in a fixed order:
    [ue_0, ue_1, ..., ue_{N-1}, server].
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: Dict[str, Any] | None = None):
        super().__init__()
        self.env = MECPhase1MAPPOEnv(env_config)
        self.n = self.env.n

        ue_obs_dim = int(np.prod(self.env.ue_obs_space.shape))
        server_obs_dim = int(np.prod(self.env.server_obs_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n * ue_obs_dim + server_obs_dim,),
            dtype=np.float32,
        )

        ue_act_dim = int(np.prod(self.env.ue_action_space.shape))
        server_act_dim = int(np.prod(self.env.server_action_space.shape))
        self.action_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n * ue_act_dim + server_act_dim,),
            dtype=np.float32,
        )

    @property
    def logs(self) -> Dict[str, Any]:
        return self.env.logs

    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        obs_list = [obs_dict[f"ue_{i}"] for i in range(self.n)]
        obs_list.append(obs_dict["server"])
        return np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in obs_list], axis=0)

    def _split_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        ue_act_dim = int(np.prod(self.env.ue_action_space.shape))
        server_act_dim = int(np.prod(self.env.server_action_space.shape))

        actions: Dict[str, np.ndarray] = {}
        cursor = 0
        for i in range(self.n):
            actions[f"ue_{i}"] = action[cursor : cursor + ue_act_dim]
            cursor += ue_act_dim
        actions["server"] = action[cursor : cursor + server_act_dim]
        return actions

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), {"__all__": infos}

    def step(self, action: np.ndarray):
        obs, rewards, terminateds, truncateds, infos = self.env.step(self._split_action(action))
        reward = float(rewards["server"])
        terminated = bool(terminateds["__all__"])
        truncated = bool(truncateds["__all__"])
        return self._flatten_obs(obs), reward, terminated, truncated, {"__all__": infos}
