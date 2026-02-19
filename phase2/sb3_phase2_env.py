from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from phase2.phase2_env import MECPhase2HierarchicalEnv


class SB3Phase2JointEnv(gym.Env):
    """
    Single-agent wrapper around the phase-2 hierarchical multi-agent environment.

    The SB3 agent outputs a joint action for coordinator, all UEs, and cluster schedulers.
    Observations are concatenated in fixed order:
    [coordinator, ue_0..ue_{N-1}, cluster_sched_0..cluster_sched_{K-1}].
    """

    metadata = {"render_modes": []}

    def __init__(self, env_config: Dict[str, Any] | None = None):
        super().__init__()
        self.env = MECPhase2HierarchicalEnv(env_config)

        self.n = self.env.n_users
        self.k = self.env.k
        self.m = self.env.m

        coord_obs_dim = int(np.prod(self.env.coordinator_obs_space.shape))
        ue_obs_dim = int(np.prod(self.env.ue_obs_space.shape))
        cluster_obs_dim = int(np.prod(self.env.cluster_sched_obs_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(coord_obs_dim + self.n * ue_obs_dim + self.k * cluster_obs_dim,),
            dtype=np.float32,
        )

        coord_act_dim = int(np.prod(self.env.coordinator_action_space.shape))
        ue_act_dim = int(np.prod(self.env.ue_action_space.shape))
        cluster_act_dim = int(np.prod(self.env.cluster_sched_action_space.shape))
        self.action_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(coord_act_dim + self.n * ue_act_dim + self.k * cluster_act_dim,),
            dtype=np.float32,
        )

    @property
    def logs(self) -> Dict[str, Any]:
        return self.env.logs

    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        parts = [obs_dict[self.env.coordinator_id]]
        parts.extend(obs_dict[uid] for uid in self.env.ue_ids)
        parts.extend(obs_dict[sid] for sid in self.env.cluster_sched_ids)
        return np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in parts], axis=0)

    def _split_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        coord_act_dim = int(np.prod(self.env.coordinator_action_space.shape))
        ue_act_dim = int(np.prod(self.env.ue_action_space.shape))
        cluster_act_dim = int(np.prod(self.env.cluster_sched_action_space.shape))

        actions: Dict[str, np.ndarray] = {}
        cursor = 0
        actions[self.env.coordinator_id] = action[cursor : cursor + coord_act_dim]
        cursor += coord_act_dim

        for uid in self.env.ue_ids:
            actions[uid] = action[cursor : cursor + ue_act_dim]
            cursor += ue_act_dim

        for sid in self.env.cluster_sched_ids:
            actions[sid] = action[cursor : cursor + cluster_act_dim]
            cursor += cluster_act_dim

        return actions

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), {"__all__": infos}

    def step(self, action: np.ndarray):
        obs, rewards, terminateds, truncateds, infos = self.env.step(self._split_action(action))
        reward = float(rewards[self.env.coordinator_id])
        terminated = bool(terminateds["__all__"])
        truncated = bool(truncateds["__all__"])
        return self._flatten_obs(obs), reward, terminated, truncated, {"__all__": infos}
