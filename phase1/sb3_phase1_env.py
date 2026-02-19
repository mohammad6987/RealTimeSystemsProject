from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

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


class SB3Phase1UEEnv(gym.Env):
    """
    UE-policy training wrapper for SB3.

    The UE PPO policy outputs a concatenated action vector for all UEs.
    The server action is provided by a separate server policy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: Dict[str, Any] | None = None,
        server_policy_getter: Callable[[], Any] | None = None,
    ):
        super().__init__()
        self.env = MECPhase1MAPPOEnv(env_config)
        self.n = self.env.n
        self.server_policy_getter = server_policy_getter

        ue_obs_dim = int(np.prod(self.env.ue_obs_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n * ue_obs_dim,),
            dtype=np.float32,
        )

        ue_act_dim = int(np.prod(self.env.ue_action_space.shape))
        self.action_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n * ue_act_dim,),
            dtype=np.float32,
        )

    @property
    def logs(self) -> Dict[str, Any]:
        return self.env.logs

    def _flatten_ue_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(obs_dict[f"ue_{i}"], dtype=np.float32).reshape(-1) for i in range(self.n)], axis=0
        )

    def _split_ue_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        ue_act_dim = int(np.prod(self.env.ue_action_space.shape))
        actions: Dict[str, np.ndarray] = {}
        cursor = 0
        for i in range(self.n):
            actions[f"ue_{i}"] = action[cursor : cursor + ue_act_dim]
            cursor += ue_act_dim
        return actions

    def _server_action(self, server_obs: np.ndarray) -> np.ndarray:
        if self.server_policy_getter is None:
            return np.zeros(self.env.server_action_space.shape, dtype=np.float32)
        policy = self.server_policy_getter()
        if policy is None:
            return np.zeros(self.env.server_action_space.shape, dtype=np.float32)
        action, _ = policy.predict(server_obs, deterministic=False)
        return np.asarray(action, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._flatten_ue_obs(obs), {"__all__": infos}

    def step(self, action: np.ndarray):
        actions = self._split_ue_action(action)
        actions["server"] = self._server_action(self.env._server_obs())
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        reward = float(rewards["server"])
        terminated = bool(terminateds["__all__"])
        truncated = bool(truncateds["__all__"])
        return self._flatten_ue_obs(obs), reward, terminated, truncated, {"__all__": infos}


class SB3Phase1ServerEnv(gym.Env):
    """
    Server-policy training wrapper for SB3.

    The server PPO policy outputs the server action; UE actions are provided
    by a shared UE policy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: Dict[str, Any] | None = None,
        ue_policy_getter: Callable[[], Any] | None = None,
    ):
        super().__init__()
        self.env = MECPhase1MAPPOEnv(env_config)
        self.n = self.env.n
        self.ue_policy_getter = ue_policy_getter

        self.observation_space = self.env.server_obs_space
        self.action_space = self.env.server_action_space

    @property
    def logs(self) -> Dict[str, Any]:
        return self.env.logs

    def _flatten_ue_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(obs_dict[f"ue_{i}"], dtype=np.float32).reshape(-1) for i in range(self.n)], axis=0
        )

    def _ue_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.ue_policy_getter is None:
            return {f"ue_{i}": np.zeros(self.env.ue_action_space.shape, dtype=np.float32) for i in range(self.n)}
        policy = self.ue_policy_getter()
        if policy is None:
            return {f"ue_{i}": np.zeros(self.env.ue_action_space.shape, dtype=np.float32) for i in range(self.n)}

        ue_obs = self._flatten_ue_obs(obs_dict)
        action, _ = policy.predict(ue_obs, deterministic=False)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        ue_act_dim = int(np.prod(self.env.ue_action_space.shape))
        actions: Dict[str, np.ndarray] = {}
        cursor = 0
        for i in range(self.n):
            actions[f"ue_{i}"] = action[cursor : cursor + ue_act_dim]
            cursor += ue_act_dim
        return actions

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, infos = self.env.reset(seed=seed, options=options)
        return obs["server"], {"__all__": infos}

    def step(self, action: np.ndarray):
        actions = self._ue_actions({f"ue_{i}": self.env._ue_obs(i) for i in range(self.n)})
        actions["server"] = np.asarray(action, dtype=np.float32)
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        reward = float(rewards["server"])
        terminated = bool(terminateds["__all__"])
        truncated = bool(truncateds["__all__"])
        return obs["server"], reward, terminated, truncated, {"__all__": infos}
