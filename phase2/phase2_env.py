from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

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
    """Softmax with minimum per-element floor, then renormalized."""
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
    """Lightweight numpy k-means to cluster users by static features."""
    n = features.shape[0]
    init_idx = rng.choice(n, size=k, replace=False)
    centers = features[init_idx].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        # Assignment step.
        d2 = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1).astype(np.int32)

        # Update step with empty-cluster recovery.
        for c in range(k):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                centers[c] = features[rng.randint(0, n)]
            else:
                centers[c] = np.mean(features[idx], axis=0)

    return labels


class MECPhase2HierarchicalEnv(MultiAgentEnv):
    """
    Phase-2 hierarchical multi-agent MEC environment.

    Agents:
    - coordinator: allocates server CPU share across clusters.
    - cluster_k: controls offload requests and per-UE priorities inside cluster k.

    Resource hierarchy:
    - Coordinator action -> cluster CPU shares phi_k (sum to 1).
    - Cluster actions -> offloading, spectrum requests, and scheduling priorities.
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
        self.cluster_ids = [f"cluster_{i}" for i in range(self.k)]
        self.agent_ids = [self.coordinator_id] + self.cluster_ids
        self.possible_agents = list(self.agent_ids)
        self.agents = list(self.agent_ids)
        self._agent_ids = set(self.agent_ids)

        # Coordinator observes per-cluster aggregates: [load, deadline, dist, f_loc, off_ratio]
        self.coordinator_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.k * 5,), dtype=np.float32)
        self.coordinator_action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.k,), dtype=np.float32)

        # Cluster controller observes a padded user-table in cluster-local order.
        # Per-row: [valid, data_norm, cycles_norm, deadline_norm, dist_norm, f_loc_norm]
        self.cluster_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.m * 6,), dtype=np.float32)
        # Cluster action packs [offload_logits, bw_logits, priority_logits] each length m.
        self.cluster_action_space = spaces.Box(low=-10.0, high=10.0, shape=(3 * self.m,), dtype=np.float32)

        self.observation_spaces = {self.coordinator_id: self.coordinator_obs_space}
        self.action_spaces = {self.coordinator_id: self.coordinator_action_space}
        for cid in self.cluster_ids:
            self.observation_spaces[cid] = self.cluster_obs_space
            self.action_spaces[cid] = self.cluster_action_space

        # UE static state and dynamic tasks.
        self.dist = np.zeros(self.n_users, dtype=np.float32)
        self.f_loc = np.zeros(self.n_users, dtype=np.float32)
        self.data_mbit = np.zeros(self.n_users, dtype=np.float32)
        self.cycles = np.zeros(self.n_users, dtype=np.float32)
        self.deadline = np.zeros(self.n_users, dtype=np.float32)

        # Cluster mapping and padding index lists.
        self.user_cluster = np.zeros(self.n_users, dtype=np.int32)
        self.cluster_users: List[np.ndarray] = []

        # Last-step offload info used in coordinator observation.
        self.last_off_ratio = np.zeros(self.k, dtype=np.float32)

        self.t = 0
        self.logs: Dict[str, Any] = {}

    @staticmethod
    def default_env_config() -> Dict[str, Any]:
        return asdict(Phase2Config())

    def _sample_static_users(self) -> None:
        """Sample static UE features and build cluster assignments."""
        self.dist = self.rng.uniform(self.cfg.dist_low, self.cfg.dist_high, size=(self.n_users,)).astype(np.float32)
        self.f_loc = self.rng.uniform(self.cfg.f_loc_low, self.cfg.f_loc_high, size=(self.n_users,)).astype(np.float32)

        # Cluster based on static user features [distance, local CPU].
        feat = np.stack(
            [
                self.dist / self.cfg.dist_high,
                self.f_loc / self.cfg.f_loc_high,
            ],
            axis=1,
        ).astype(np.float32)

        self.user_cluster = run_kmeans(feat, self.k, self.cfg.kmeans_iters, self.rng)
        self.cluster_users = [np.where(self.user_cluster == c)[0] for c in range(self.k)]

        # Guard against empty clusters after k-means by moving random users.
        for c in range(self.k):
            if self.cluster_users[c].size == 0:
                u = self.rng.randint(0, self.n_users)
                old_c = int(self.user_cluster[u])
                self.user_cluster[u] = c
                self.cluster_users = [np.where(self.user_cluster == j)[0] for j in range(self.k)]
                if self.cluster_users[old_c].size == 0:
                    # Ensure donor cluster is not left empty.
                    v = (u + 1) % self.n_users
                    self.user_cluster[v] = old_c
                    self.cluster_users = [np.where(self.user_cluster == j)[0] for j in range(self.k)]

    def _sample_tasks(self) -> None:
        """Sample one fresh task per user for current slot."""
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

    def _coordinator_obs(self) -> np.ndarray:
        """Aggregated cluster-level state for top-level allocation decisions."""
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

    def _cluster_obs(self, c: int) -> np.ndarray:
        """Padded per-user cluster observation with a validity mask."""
        users = self.cluster_users[c]
        obs = np.zeros((self.m, 6), dtype=np.float32)

        limit = min(users.size, self.m)
        for j in range(limit):
            u = int(users[j])
            obs[j, 0] = 1.0
            obs[j, 1] = self.data_mbit[u] / self.cfg.data_mbit_high
            obs[j, 2] = (self.cycles[u] / 1e9) / self.cfg.cycles_g_high
            obs[j, 3] = self.deadline[u] / self.cfg.deadline_high
            obs[j, 4] = self.dist[u] / self.cfg.dist_high
            obs[j, 5] = self.f_loc[u] / self.cfg.f_loc_high

        return obs.reshape(-1).astype(np.float32)

    def _obs_dict(self) -> Dict[str, np.ndarray]:
        obs = {self.coordinator_id: self._coordinator_obs()}
        for c, cid in enumerate(self.cluster_ids):
            obs[cid] = self._cluster_obs(c)
        return obs

    def reset(self, *, seed=None, options=None):
        """Reset full hierarchical system and return observations for all agents."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.t = 0
        self._sample_static_users()
        self._sample_tasks()
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
        # 1) Coordinator allocates normalized CPU share to each cluster.
        coord_logits = np.asarray(action_dict[self.coordinator_id], dtype=np.float32)
        phi = stable_softmax(coord_logits)
        phi = np.maximum(phi, 1e-3)
        phi = phi / np.sum(phi)

        # 2) Decode cluster actions into user-level offloading/spectrum/priorities.
        offload = np.zeros(self.n_users, dtype=np.int32)
        bw_req_logits = np.full(self.n_users, -1e9, dtype=np.float32)
        priority_logits = np.full(self.n_users, -1e9, dtype=np.float32)

        for c, cid in enumerate(self.cluster_ids):
            act = np.asarray(action_dict[cid], dtype=np.float32)
            off_l = act[: self.m]
            bw_l = act[self.m : 2 * self.m]
            pr_l = act[2 * self.m : 3 * self.m]

            users = self.cluster_users[c]
            n_local = min(users.size, self.m)
            if n_local == 0:
                self.last_off_ratio[c] = 0.0
                continue

            loc_off = (off_l[:n_local] > 0.0).astype(np.int32)
            mapped_users = users[:n_local]
            offload[mapped_users] = loc_off
            bw_req_logits[mapped_users] = bw_l[:n_local]
            priority_logits[mapped_users] = pr_l[:n_local]

            self.last_off_ratio[c] = float(np.mean(loc_off))

        off_idx = np.where(offload == 1)[0]

        # 3) Allocate global uplink spectrum among all offloaded users.
        theta = np.zeros(self.n_users, dtype=np.float32)
        if off_idx.size > 0:
            theta_off = floor_softmax(bw_req_logits[off_idx], self.cfg.theta_min)
            theta[off_idx] = theta_off

        # 4) Compute transmission and local execution costs.
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

        # 5) Cluster-level server scheduling under coordinator CPU share.
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

            f_cluster = float(phi[c] * self.cfg.f_mec)
            f_cluster = max(f_cluster, 1e6)

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

        # 6) QoS and energy accounting (total + per-cluster).
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

        # Distribute server energy to clusters by their share of server execution time.
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

        # 7) Move to next slot.
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
