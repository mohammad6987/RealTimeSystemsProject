from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from phase2.phase2_config import Phase2Config


def qos_value(delay: float, deadline: float) -> float:
    """Piecewise QoS function used in both project phases."""
    if delay <= deadline:
        return 1.0
    if delay < 2.0 * deadline:
        return 1.0 - (delay - deadline) / deadline
    return 0.0


def stable_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax helper for allocation logits."""
    z = x.astype(np.float64)
    z = z - np.max(z)
    exp_z = np.exp(z)
    return (exp_z / (np.sum(exp_z) + 1e-12)).astype(np.float32)


def floor_softmax(x: np.ndarray, floor: float) -> np.ndarray:
    """Softmax with a minimum floor per element, then renormalized."""
    if x.size == 0:
        return x.astype(np.float32)
    p = stable_softmax(x)
    floor = float(np.clip(floor, 0.0, 1.0 / max(1, x.size)))
    if floor <= 0.0:
        return p
    p = floor + (1.0 - floor * x.size) * p
    p = p / (np.sum(p) + 1e-12)
    return p.astype(np.float32)


def run_kmeans(features: np.ndarray, k: int, iters: int, rng: np.random.RandomState) -> np.ndarray:
    """Lightweight NumPy k-means for cluster assignment."""
    n = features.shape[0]
    init_idx = rng.choice(n, size=k, replace=False)
    centers = features[init_idx].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        d2 = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1).astype(np.int32)

        for c in range(k):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                centers[c] = features[rng.randint(0, n)]
            else:
                centers[c] = np.mean(features[idx], axis=0)

    return labels


class MECPhase2HierarchicalEnv(MultiAgentEnv):
    """
    Phase-2 hierarchical environment (UE-driven offloading).

    Agents:
    - coordinator: global server resource allocator across clusters.
    - ue_i: each UE decides offload + bandwidth request (same idea as phase 1).
    - cluster_sched_j: one scheduler/coordinator per cluster, decides priority order.

    Hierarchy:
    - UE requests -> cluster scheduler order -> execution under coordinator CPU shares.
    """

    def __init__(self, env_config: Dict[str, Any] | None = None):
        super().__init__()
        env_config = env_config or {}
        self.cfg = Phase2Config(**env_config)

        self.n_users = self.cfg.n_users
        self.k = self.cfg.n_clusters
        self.max_steps = self.cfg.max_steps
        self.m = self.cfg.max_users_per_cluster

        self.rng = np.random.RandomState(self.cfg.seed)

        self.coordinator_id = "coordinator"
        self.ue_ids = [f"ue_{i}" for i in range(self.n_users)]
        self.cluster_sched_ids = [f"cluster_sched_{i}" for i in range(self.k)]

        self.agent_ids = [self.coordinator_id] + self.ue_ids + self.cluster_sched_ids
        self.possible_agents = list(self.agent_ids)
        self.agents = list(self.agent_ids)
        self._agent_ids = set(self.agent_ids)

        # UE interface: same decision semantics as phase 1.
        self.ue_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.ue_action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)

        # Global coordinator interface.
        self.coordinator_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.k * 5,), dtype=np.float32)
        self.coordinator_action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.k,), dtype=np.float32)

        # Cluster scheduler gets request-rich cluster table and outputs priorities.
        # Per-row: [valid, data_norm, cycles_norm, deadline_norm, dist_norm, req_off, req_bw_logit]
        self.cluster_sched_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.m * 7,), dtype=np.float32)
        self.cluster_sched_action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.m,), dtype=np.float32)

        self.observation_spaces = {self.coordinator_id: self.coordinator_obs_space}
        self.action_spaces = {self.coordinator_id: self.coordinator_action_space}

        for uid in self.ue_ids:
            self.observation_spaces[uid] = self.ue_obs_space
            self.action_spaces[uid] = self.ue_action_space

        for sid in self.cluster_sched_ids:
            self.observation_spaces[sid] = self.cluster_sched_obs_space
            self.action_spaces[sid] = self.cluster_sched_action_space

        # UE static + dynamic state.
        self.dist = np.zeros(self.n_users, dtype=np.float32)
        self.f_loc = np.zeros(self.n_users, dtype=np.float32)
        self.data_mbit = np.zeros(self.n_users, dtype=np.float32)
        self.cycles = np.zeros(self.n_users, dtype=np.float32)
        self.deadline = np.zeros(self.n_users, dtype=np.float32)

        # Cluster mapping.
        self.user_cluster = np.zeros(self.n_users, dtype=np.int32)
        self.cluster_users: List[np.ndarray] = []

        # Last-step UE requests for coordinator/scheduler observations.
        self.req_off = np.zeros(self.n_users, dtype=np.float32)
        self.req_bw_logit = np.zeros(self.n_users, dtype=np.float32)
        self.last_off_ratio = np.zeros(self.k, dtype=np.float32)

        self.t = 0
        self.logs: Dict[str, Any] = {}

    @staticmethod
    def default_env_config() -> Dict[str, Any]:
        return asdict(Phase2Config())

    def _sample_static_users(self) -> None:
        """Sample static UE features and cluster assignments."""
        self.dist = self.rng.uniform(self.cfg.dist_low, self.cfg.dist_high, size=(self.n_users,)).astype(np.float32)
        self.f_loc = self.rng.uniform(self.cfg.f_loc_low, self.cfg.f_loc_high, size=(self.n_users,)).astype(np.float32)

        feat = np.stack(
            [
                self.dist / self.cfg.dist_high,
                self.f_loc / self.cfg.f_loc_high,
            ],
            axis=1,
        ).astype(np.float32)

        self.user_cluster = run_kmeans(feat, self.k, self.cfg.kmeans_iters, self.rng)
        self.cluster_users = [np.where(self.user_cluster == c)[0] for c in range(self.k)]

        # Ensure no empty cluster.
        for c in range(self.k):
            if self.cluster_users[c].size == 0:
                u = self.rng.randint(0, self.n_users)
                self.user_cluster[u] = c
                self.cluster_users = [np.where(self.user_cluster == j)[0] for j in range(self.k)]

    def _sample_tasks(self) -> None:
        """Sample one task per UE for current slot."""
        self.data_mbit = self.rng.uniform(self.cfg.data_mbit_low, self.cfg.data_mbit_high, size=(self.n_users,)).astype(
            np.float32
        )
        cycles_g = self.rng.uniform(self.cfg.cycles_g_low, self.cfg.cycles_g_high, size=(self.n_users,)).astype(np.float32)
        self.cycles = (cycles_g * 1e9).astype(np.float32)
        self.deadline = self.rng.uniform(self.cfg.deadline_low, self.cfg.deadline_high, size=(self.n_users,)).astype(np.float32)

    def _channel_gain(self, d: float) -> float:
        return float(1.0 / (max(d, 1.0) ** 2))

    def _uplink_rate(self, theta: float, dist: float) -> float:
        b = max(theta * self.cfg.bandwidth_hz, 1e-9)
        snr = (self.cfg.tx_power_w * self._channel_gain(dist)) / (b * self.cfg.noise_psd)
        return b * np.log2(1.0 + snr + 1e-12)

    def _ue_obs(self, u: int) -> np.ndarray:
        """UE observation vector (phase-1 style)."""
        return np.array(
            [
                self.data_mbit[u],
                self.cycles[u] / 1e9,
                self.deadline[u],
                self.dist[u],
                self.f_loc[u] / 1e9,
            ],
            dtype=np.float32,
        )

    def _coordinator_obs(self) -> np.ndarray:
        """Cluster aggregates for global resource allocation."""
        obs = np.zeros((self.k, 5), dtype=np.float32)
        for c in range(self.k):
            users = self.cluster_users[c]
            if users.size == 0:
                continue
            obs[c, 0] = float(np.mean(self.cycles[users] / 1e9) / self.cfg.cycles_g_high)
            obs[c, 1] = float(np.mean(self.deadline[users]) / self.cfg.deadline_high)
            obs[c, 2] = float(np.mean(self.dist[users]) / self.cfg.dist_high)
            obs[c, 3] = float(np.mean(self.f_loc[users]) / self.cfg.f_loc_high)
            obs[c, 4] = float(self.last_off_ratio[c])
        return obs.reshape(-1).astype(np.float32)

    def _cluster_sched_obs(self, c: int) -> np.ndarray:
        """Scheduler observation with UE requests in this cluster."""
        users = self.cluster_users[c]
        obs = np.zeros((self.m, 7), dtype=np.float32)

        n_local = min(users.size, self.m)
        for j in range(n_local):
            u = int(users[j])
            obs[j, 0] = 1.0
            obs[j, 1] = self.data_mbit[u] / self.cfg.data_mbit_high
            obs[j, 2] = (self.cycles[u] / 1e9) / self.cfg.cycles_g_high
            obs[j, 3] = self.deadline[u] / self.cfg.deadline_high
            obs[j, 4] = self.dist[u] / self.cfg.dist_high
            obs[j, 5] = self.req_off[u]
            obs[j, 6] = self.req_bw_logit[u] / 10.0

        return obs.reshape(-1).astype(np.float32)

    def _obs_dict(self) -> Dict[str, np.ndarray]:
        obs = {self.coordinator_id: self._coordinator_obs()}
        for u, uid in enumerate(self.ue_ids):
            obs[uid] = self._ue_obs(u)
        for c, sid in enumerate(self.cluster_sched_ids):
            obs[sid] = self._cluster_sched_obs(c)
        return obs

    def reset(self, *, seed=None, options=None):
        """Reset and return initial observations for all agents."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.t = 0
        self._sample_static_users()
        self._sample_tasks()
        self.req_off[:] = 0.0
        self.req_bw_logit[:] = 0.0
        self.last_off_ratio[:] = 0.0

        self.logs = {
            "sum_qos": [],
            "sum_energy": [],
            "cluster_qos": [[] for _ in range(self.k)],
            "cluster_energy": [[] for _ in range(self.k)],
            "schedule": [],
        }

        obs = self._obs_dict()
        infos = {aid: {} for aid in obs}
        return obs, infos

    def step(self, action_dict: Dict[str, np.ndarray]):
        """Apply one hierarchical action and advance one slot."""
        # 1) UE decisions (offload + bw request logits).
        offload = np.zeros(self.n_users, dtype=np.int32)
        bw_logits = np.zeros(self.n_users, dtype=np.float32)
        for u, uid in enumerate(self.ue_ids):
            a = np.asarray(action_dict[uid], dtype=np.float32)
            offload[u] = 1 if a[0] > 0.0 else 0
            bw_logits[u] = a[1]

        self.req_off = offload.astype(np.float32)
        self.req_bw_logit = bw_logits.copy()

        for c in range(self.k):
            users = self.cluster_users[c]
            self.last_off_ratio[c] = float(np.mean(offload[users])) if users.size > 0 else 0.0

        # 2) Global coordinator allocates cluster compute shares.
        coord_logits = np.asarray(action_dict[self.coordinator_id], dtype=np.float32)
        phi = stable_softmax(coord_logits)
        phi = np.maximum(phi, 1e-3)
        phi = phi / (np.sum(phi) + 1e-12)

        # 3) Cluster scheduler priorities (one scheduler per cluster).
        priority_logits = np.full(self.n_users, -1e9, dtype=np.float32)
        for c, sid in enumerate(self.cluster_sched_ids):
            pr = np.asarray(action_dict[sid], dtype=np.float32)
            users = self.cluster_users[c]
            n_local = min(users.size, self.m)
            if n_local > 0:
                priority_logits[users[:n_local]] = pr[:n_local]

        # 4) Global spectrum allocation for all offloaded UEs.
        off_idx = np.where(offload == 1)[0]
        theta = np.zeros(self.n_users, dtype=np.float32)
        if off_idx.size > 0:
            theta_off = floor_softmax(bw_logits[off_idx], self.cfg.theta_min)
            theta[off_idx] = theta_off

        # 5) Comm + local costs.
        tx_time = np.zeros(self.n_users, dtype=np.float32)
        tx_energy = np.zeros(self.n_users, dtype=np.float32)
        for u in off_idx:
            rate = self._uplink_rate(float(theta[u]), float(self.dist[u]))
            bits = float(self.data_mbit[u] * 1e6)
            tx_time[u] = bits / max(rate, 1e-9)
            tx_energy[u] = self.cfg.tx_power_w * tx_time[u]

        local_time = np.zeros(self.n_users, dtype=np.float32)
        local_energy = np.zeros(self.n_users, dtype=np.float32)
        for u in range(self.n_users):
            if offload[u] == 0:
                local_time[u] = self.cycles[u] / self.f_loc[u]
                local_energy[u] = self.cfg.kappa * self.cycles[u] * (self.f_loc[u] ** 2)

        # 6) Cluster-level scheduling with cluster-specific CPU budgets.
        completion = np.zeros(self.n_users, dtype=np.float32)
        exec_time = np.zeros(self.n_users, dtype=np.float32)

        slot_sched: List[Dict[str, Any]] = []
        per_cluster_qos = np.zeros(self.k, dtype=np.float32)
        per_cluster_energy = np.zeros(self.k, dtype=np.float32)

        for c in range(self.k):
            users = self.cluster_users[c]
            if users.size == 0:
                slot_sched.append({"cluster": c, "order": [], "start": {}, "finish": {}})
                continue

            f_cluster = max(float(phi[c] * self.cfg.f_mec), 1e6)
            off_c = [int(u) for u in users if offload[int(u)] == 1]

            starts: Dict[int, float] = {}
            finishes: Dict[int, float] = {}

            if off_c:
                order = sorted(off_c, key=lambda u: float(priority_logits[u]), reverse=True)
                clk = 0.0
                for u in order:
                    exe = float(self.cycles[u] / f_cluster)
                    exec_time[u] = exe
                    st = max(clk, float(tx_time[u]))
                    ft = st + exe
                    starts[u] = st
                    finishes[u] = ft
                    completion[u] = ft
                    clk = ft
            else:
                order = []

            for u in users:
                uu = int(u)
                if offload[uu] == 0:
                    completion[uu] = local_time[uu]

            slot_sched.append({"cluster": c, "order": order, "start": starts, "finish": finishes})

        # 7) QoS + energy accounting.
        qos = np.zeros(self.n_users, dtype=np.float32)
        for u in range(self.n_users):
            qos[u] = qos_value(float(completion[u]), float(self.deadline[u]))

        ue_energy = local_energy + tx_energy
        server_energy = self.cfg.p_mec * float(np.sum(exec_time[off_idx])) if off_idx.size > 0 else 0.0

        for c in range(self.k):
            users = self.cluster_users[c]
            if users.size == 0:
                continue
            per_cluster_qos[c] = float(np.sum(qos[users]))
            per_cluster_energy[c] = float(np.sum(ue_energy[users]))

        # Split server energy across clusters proportional to server execution use.
        exec_by_cluster = np.zeros(self.k, dtype=np.float32)
        for c in range(self.k):
            users = self.cluster_users[c]
            if users.size > 0:
                exec_by_cluster[c] = float(np.sum(exec_time[users]))
        exec_total = float(np.sum(exec_by_cluster))
        if exec_total > 0.0:
            per_cluster_energy += server_energy * (exec_by_cluster / exec_total)

        total_qos = float(np.sum(qos))
        total_energy = float(np.sum(ue_energy) + server_energy)
        reward = self.cfg.w_qos * total_qos - self.cfg.w_energy * total_energy

        rewards = {aid: float(reward) for aid in self.agent_ids}

        self.logs["sum_qos"].append(total_qos)
        self.logs["sum_energy"].append(total_energy)
        for c in range(self.k):
            self.logs["cluster_qos"][c].append(float(per_cluster_qos[c]))
            self.logs["cluster_energy"][c].append(float(per_cluster_energy[c]))
        self.logs["schedule"].append({"slot": self.t, "clusters": slot_sched})

        # Next slot.
        self._sample_tasks()
        self.t += 1

        done = self.t >= self.max_steps
        terminateds = {aid: done for aid in self.agent_ids}
        terminateds["__all__"] = done
        truncateds = {aid: False for aid in self.agent_ids}
        truncateds["__all__"] = False

        obs = self._obs_dict()
        infos = {aid: {} for aid in self.agent_ids}
        infos[self.coordinator_id] = {
            "sum_qos": total_qos,
            "sum_energy": total_energy,
            "offloaded_count": int(off_idx.size),
        }

        return obs, rewards, terminateds, truncateds, infos
