from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from phase1.mec_phase1_config import Phase1Config


def qos_value(delay: float, deadline: float) -> float:
    """
    Piecewise QoS function from the project statement.

    Returns:
    - 1.0 for on-time completion,
    - linear decay between D and 2D,
    - 0.0 after 2D.
    """
    if delay <= deadline:
        return 1.0
    if delay < 2.0 * deadline:
        return 1.0 - (delay - deadline) / deadline
    return 0.0


def stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax used for spectrum-share normalization."""
    z = x.astype(np.float64)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return (exp_z / (np.sum(exp_z) + 1e-12)).astype(np.float32)


class MECPhase1MAPPOEnv(MultiAgentEnv):
    """
    Phase-1 environment:
    - UE agents decide offload + spectrum request logit.
    - Server agent decides learned scheduling priorities.

    Step dynamics:
    1) Decode UE offloading and bandwidth requests.
    2) Allocate uplink spectrum among offloaded UEs.
    3) Compute uplink delays/energies and local execution costs.
    4) Server orders offloaded tasks using learned priorities.
    5) Compute completion times, QoS, energy, and global reward.
    """

    def __init__(self, env_config: Dict[str, Any] | None = None):
        super().__init__()
        env_config = env_config or {}
        self.cfg = Phase1Config(**env_config)
        self.n = self.cfg.n_ue
        self.max_steps = self.cfg.max_steps

        # Local RNG keeps environment stochastic but reproducible.
        self.rng = np.random.RandomState(self.cfg.seed)
        self.agent_ids = [f"ue_{i}" for i in range(self.n)] + ["server"]
        # RLlib compatibility fields for multi-agent pre-checks.
        self.possible_agents = list(self.agent_ids)
        self.agents = list(self.agent_ids)
        self._agent_ids = set(self.agent_ids)

        # UE obs: [data_mbit, cycles_g, deadline_s, dist_m, f_loc_ghz]
        self.ue_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        # UE action: [offload_logit, spectrum_request_logit]
        self.ue_action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)

        # Server obs: flattened per-UE table
        # [data_norm, cycles_norm, deadline_norm, dist_norm, req_offload, req_spectrum_share]
        self.server_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n * 6,), dtype=np.float32)
        # Server action: one priority score per UE
        self.server_action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.n,), dtype=np.float32)

        self.observation_spaces = {f"ue_{i}": self.ue_obs_space for i in range(self.n)}
        self.observation_spaces["server"] = self.server_obs_space

        self.action_spaces = {f"ue_{i}": self.ue_action_space for i in range(self.n)}
        self.action_spaces["server"] = self.server_action_space

        # Time-slot index inside current episode.
        self.t = 0
        # Static/semi-static UE properties sampled on reset.
        self.dist = np.zeros(self.n, dtype=np.float32)
        self.f_loc = np.zeros(self.n, dtype=np.float32)
        # Per-slot task properties sampled every step.
        self.data_mbit = np.zeros(self.n, dtype=np.float32)
        self.cycles = np.zeros(self.n, dtype=np.float32)
        self.deadline = np.zeros(self.n, dtype=np.float32)

        # Last-step UE requests (included in server observation).
        self.req_offload = np.zeros(self.n, dtype=np.float32)
        self.req_spectrum = np.zeros(self.n, dtype=np.float32)

        # Episode logs used later for plotting and analysis.
        self.logs: Dict[str, Any] = {}

    @staticmethod
    def default_env_config() -> Dict[str, Any]:
        """Return a plain-dict default config for easy external initialization."""
        return asdict(Phase1Config())

    def _sample_ues(self) -> None:
        """Sample UE distances and local CPU capacities for the episode."""
        self.dist = self.rng.uniform(self.cfg.dist_low, self.cfg.dist_high, size=(self.n,)).astype(np.float32)
        self.f_loc = self.rng.uniform(self.cfg.f_loc_low, self.cfg.f_loc_high, size=(self.n,)).astype(np.float32)

    def _sample_tasks(self) -> None:
        """Sample one task per UE for the current time slot."""
        self.data_mbit = self.rng.uniform(
            self.cfg.data_mbit_low, self.cfg.data_mbit_high, size=(self.n,)
        ).astype(np.float32)
        cycles_g = self.rng.uniform(self.cfg.cycles_g_low, self.cfg.cycles_g_high, size=(self.n,)).astype(np.float32)
        self.cycles = (cycles_g * 1e9).astype(np.float32)
        self.deadline = self.rng.uniform(
            self.cfg.deadline_low, self.cfg.deadline_high, size=(self.n,)
        ).astype(np.float32)

    def _channel_gain(self, d: float) -> float:
        """Simple path-loss proxy g ~ 1/d^2 used by the communication model."""
        return float(1.0 / (max(d, 1.0) ** 2))

    def _uplink_rate(self, theta: float, dist: float) -> float:
        """Compute UE uplink rate using a Shannon-like expression."""
        bandwidth = max(theta * self.cfg.bandwidth_hz, 1e-9)
        gain = self._channel_gain(dist)
        snr = (self.cfg.tx_power_w * gain) / (bandwidth * self.cfg.noise_psd)
        return bandwidth * np.log2(1.0 + snr + 1e-12)

    def _ue_obs(self, i: int) -> np.ndarray:
        """Build observation vector for one UE agent."""
        return np.array(
            [
                self.data_mbit[i],
                self.cycles[i] / 1e9,
                self.deadline[i],
                self.dist[i],
                self.f_loc[i] / 1e9,
            ],
            dtype=np.float32,
        )

    def _server_obs(self) -> np.ndarray:
        """
        Build flattened server observation.

        Per UE row includes normalized task stats + requested decisions from UEs.
        Flattening is used to satisfy RLlib default encoder expectations.
        """
        obs = np.zeros((self.n, 6), dtype=np.float32)
        for i in range(self.n):
            obs[i, 0] = self.data_mbit[i] / self.cfg.data_mbit_high
            obs[i, 1] = (self.cycles[i] / 1e9) / self.cfg.cycles_g_high
            obs[i, 2] = self.deadline[i] / self.cfg.deadline_high
            obs[i, 3] = self.dist[i] / self.cfg.dist_high
            obs[i, 4] = self.req_offload[i]
            obs[i, 5] = self.req_spectrum[i]
        return obs.reshape(-1).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """Start a new episode and return initial observations for all agents."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.t = 0
        self._sample_ues()
        self._sample_tasks()
        self.req_offload[:] = 0.0
        self.req_spectrum[:] = 0.0

        # Each list stores one entry per environment step.
        self.logs = {
            "sum_qos": [],
            "sum_energy": [],
            "per_ue_qos": [[] for _ in range(self.n)],
            "per_ue_energy": [[] for _ in range(self.n)],
            "scheduling": [],
        }

        obs = {f"ue_{i}": self._ue_obs(i) for i in range(self.n)}
        obs["server"] = self._server_obs()
        infos = {aid: {} for aid in obs}
        return obs, infos

    def step(self, action_dict: Dict[str, np.ndarray]):
        """
        Apply one multi-agent action and advance the environment by one slot.

        Args:
            action_dict: UE actions and server priority action.
        Returns:
            Standard RLlib MA tuple:
            (obs, rewards, terminateds, truncateds, infos)
        """
        # 1) Decode UE decisions from logits.
        offload = np.zeros(self.n, dtype=np.int32)
        spectrum_logits = np.zeros(self.n, dtype=np.float32)

        for i in range(self.n):
            ue_action = np.asarray(action_dict[f"ue_{i}"], dtype=np.float32)
            offload[i] = 1 if ue_action[0] > 0.0 else 0
            spectrum_logits[i] = ue_action[1]

        self.req_offload = offload.astype(np.float32)
        off_idx = np.where(offload == 1)[0]

        # 2) Convert requested spectrum logits to valid shares for offloaders.
        theta = np.zeros(self.n, dtype=np.float32)
        if off_idx.size > 0:
            theta[off_idx] = stable_softmax(spectrum_logits[off_idx])
        self.req_spectrum = theta.copy()

        # 3) Offloading transmission costs (delay and energy).
        tx_time = np.zeros(self.n, dtype=np.float32)
        tx_energy = np.zeros(self.n, dtype=np.float32)
        for i in off_idx:
            rate_bps = self._uplink_rate(float(theta[i]), float(self.dist[i]))
            bits = float(self.data_mbit[i] * 1e6)
            tx_time[i] = bits / max(rate_bps, 1e-9)
            tx_energy[i] = self.cfg.tx_power_w * tx_time[i]

        # 4) Local execution costs for UEs that do not offload.
        local_time = np.zeros(self.n, dtype=np.float32)
        local_energy = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            if offload[i] == 0:
                local_time[i] = self.cycles[i] / self.f_loc[i]
                local_energy[i] = self.cfg.kappa * self.cycles[i] * (self.f_loc[i] ** 2)

        # 5) Server execution time for offloaded tasks.
        priorities = np.asarray(action_dict["server"], dtype=np.float32)
        exec_time = np.zeros(self.n, dtype=np.float32)
        for i in off_idx:
            exec_time[i] = self.cycles[i] / self.cfg.f_mec

        completion = np.zeros(self.n, dtype=np.float32)
        starts: Dict[int, float] = {}
        finishes: Dict[int, float] = {}

        if off_idx.size > 0:
            # Server policy controls schedule order through descending priority.
            order = sorted(list(off_idx), key=lambda i: float(priorities[i]), reverse=True)
            server_clock = 0.0
            for i in order:
                # Arrival-aware scheduling: task can only start after upload completes.
                release = float(tx_time[i])
                start_t = max(server_clock, release)
                finish_t = start_t + float(exec_time[i])
                starts[int(i)] = start_t
                finishes[int(i)] = finish_t
                server_clock = finish_t
                completion[i] = finish_t
        else:
            order = []

        for i in range(self.n):
            if offload[i] == 0:
                completion[i] = local_time[i]

        # 6) QoS per UE from completion vs deadline.
        qos = np.zeros(self.n, dtype=np.float32)
        for i in range(self.n):
            qos[i] = qos_value(float(completion[i]), float(self.deadline[i]))

        # 7) Global energy accounting: UE local/tx + server compute energy.
        server_energy = self.cfg.p_mec * float(np.sum(exec_time[off_idx])) if off_idx.size > 0 else 0.0
        ue_energy = local_energy + tx_energy

        # 8) Shared cooperative reward for all agents.
        total_qos = float(np.sum(qos))
        total_energy = float(np.sum(ue_energy) + server_energy)
        global_reward = self.cfg.w_qos * total_qos - self.cfg.w_energy * total_energy

        rewards = {aid: float(global_reward) for aid in self.agent_ids}

        # 9) Log metrics for plotting/reporting.
        self.logs["sum_qos"].append(total_qos)
        self.logs["sum_energy"].append(total_energy)
        for i in range(self.n):
            self.logs["per_ue_qos"][i].append(float(qos[i]))
            self.logs["per_ue_energy"][i].append(float(ue_energy[i]))
        self.logs["scheduling"].append(
            {
                "slot": self.t,
                "order": [int(i) for i in order],
                "start": starts,
                "finish": finishes,
            }
        )

        # 10) Prepare next slot.
        self._sample_tasks()
        self.t += 1

        done = self.t >= self.max_steps
        terminateds = {aid: done for aid in self.agent_ids}
        terminateds["__all__"] = done
        truncateds = {aid: False for aid in self.agent_ids}
        truncateds["__all__"] = False

        obs = {f"ue_{i}": self._ue_obs(i) for i in range(self.n)}
        obs["server"] = self._server_obs()

        infos = {aid: {} for aid in self.agent_ids}
        infos["server"] = {
            "sum_qos": total_qos,
            "sum_energy": total_energy,
            "offloaded_count": int(off_idx.size),
        }

        return obs, rewards, terminateds, truncateds, infos
