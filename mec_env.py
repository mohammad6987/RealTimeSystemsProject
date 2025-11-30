# mec_env.py
"""
Agents:
 - ue_0, ue_1, ..., ue_{N-1} (UE agents: propose offload + requested beta/theta)
 - server (single server agent: accept mask + final beta/theta allocation)

Environment implements:
 - per-slot task generation
 - local execution time/energy
 - offloading transmission time/energy and MEC execution time/energy
 - EDF scheduling of accepted offloaded tasks
 - QoS piecewise function (exact from Project20.pdf)
 - per-agent shaped rewards and global reward
"""

from typing import Dict, Any, Tuple
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

def qos_function(delay: float, deadline: float) -> float:
    if delay <= deadline:
        return 1.0
    elif delay < 2.0 * deadline:
        return 1.0 - (delay - deadline) / deadline
    else:
        return 0.0

class MECEnv(MultiAgentEnv):
    def __init__(self, env_config: Dict = None):
        env_config = env_config or {}
        self.N = int(env_config.get("N", 5))
        self.seed = int(env_config.get("seed", 0))
        self.rng = np.random.RandomState(self.seed)
        self.slot_time = float(env_config.get("slot_time", 1.0))
        self.max_steps = int(env_config.get("max_steps", 200))
        self.step_count = 0

        self.W = 20e6                      # Hz (20 MHz)
        self.f_mec = 6e9                   # cycles/sec (6 GHz)
        self.P_mec = 5.0                   # W
        self.N0 = 1e-9                     # noise power density (simplified)
        self.p_idle = np.full(self.N, 0.1) # W idle UE power
        self.p_tx = np.full(self.N, 0.5)   # W transmit power (UE)

        # Per-episode UE physical capabilities (sampled once per reset)
        self.f_loc = np.zeros(self.N)      # local CPU freq (Hz)
        self.dist = np.zeros(self.N)       # distance (m)
        self.g_i = np.zeros(self.N)        # channel gain (simple pathloss)

        # Per-slot tasks (sampled each step)
        self.s = np.zeros(self.N)  # bits
        self.c = np.zeros(self.N)  # cycles
        self.D = np.zeros(self.N)  # seconds (deadlines)

        # Build action & observation spaces per agent:
        ue_action_space = spaces.Dict({
            "offload": spaces.Discrete(2),
            "beta": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "theta": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        server_action_space = spaces.Dict({
            "accept": spaces.MultiBinary(self.N),
            "beta_alloc": spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
            "theta_alloc": spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
        })

        ue_obs_space = spaces.Dict({
            "s": spaces.Box(low=0.0, high=1e9, shape=(1,), dtype=np.float32),
            "c": spaces.Box(low=0.0, high=1e12, shape=(1,), dtype=np.float32),
            "D": spaces.Box(low=0.0, high=1e4, shape=(1,), dtype=np.float32),
            "f_loc": spaces.Box(low=0.0, high=1e10, shape=(1,), dtype=np.float32),
            "p_tx": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
            "g": spaces.Box(low=0.0, high=1e3, shape=(1,), dtype=np.float32),
            "avail_cpu_frac": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "avail_bw_frac": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        server_obs_space = spaces.Dict({
            "req_offload": spaces.MultiBinary(self.N),
            "req_beta": spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
            "req_theta": spaces.Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
            "s": spaces.Box(low=0.0, high=1e9, shape=(self.N,), dtype=np.float32),
            "c": spaces.Box(low=0.0, high=1e12, shape=(self.N,), dtype=np.float32),
            "D": spaces.Box(low=0.0, high=1e4, shape=(self.N,), dtype=np.float32),
            "g": spaces.Box(low=0.0, high=1e3, shape=(self.N,), dtype=np.float32),
            "avail_cpu_frac": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "avail_bw_frac": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        # store spaces in dicts for trainer access
        self.observation_spaces = {f"ue_{i}": ue_obs_space for i in range(self.N)}
        self.observation_spaces["server"] = server_obs_space
        self.action_spaces = {f"ue_{i}": ue_action_space for i in range(self.N)}
        self.action_spaces["server"] = server_action_space

        # logging
        self.episode_metrics = None
        self.reset()

    # helper to sample per-episode UE capabilities
    def _sample_episode_parameters(self):
        # local CPU freq in Hz (0.5-1.5 GHz)
        self.f_loc = self.rng.uniform(0.5e9, 1.5e9, size=(self.N,))
        # distances in meters
        self.dist = self.rng.uniform(10.0, 50.0, size=(self.N,))
        # simple pathloss channel gain (1/d^2)
        self.g_i = 1.0 / (np.maximum(self.dist, 1.0) ** 2)

    def reset(self) -> Dict[str, Any]:
        self.step_count = 0
        self._sample_episode_parameters()
        self.episode_metrics = {
            "per_step_energy": [],
            "per_step_qos": [],
            "per_ue_energy": [[] for _ in range(self.N)],
            "per_ue_qos": [[] for _ in range(self.N)],
            "schedule": []
        }
        self._sample_slot_tasks()
        return self._get_obs()

    def _sample_slot_tasks(self):
        # sizes in bits (16-80 Mbit -> convert to bits)
        self.s = self.rng.uniform(16.0e6, 80.0e6, size=(self.N,))
        # required cycles (1.5-3.5 Gcycles -> convert to cycles)
        self.c = self.rng.uniform(1.5e9, 3.5e9, size=(self.N,))
        # deadlines
        self.D = self.rng.uniform(2.0, 4.0, size=(self.N,))

    def _get_obs(self) -> Dict[str, Any]:
        avail_cpu_frac = np.array([1.0], dtype=np.float32)
        avail_bw_frac = np.array([1.0], dtype=np.float32)
        obs = {}
        for i in range(self.N):
            obs[f"ue_{i}"] = {
                "s": np.array([self.s[i]], dtype=np.float32),
                "c": np.array([self.c[i]], dtype=np.float32),
                "D": np.array([self.D[i]], dtype=np.float32),
                "f_loc": np.array([self.f_loc[i]], dtype=np.float32),
                "p_tx": np.array([self.p_tx[i]], dtype=np.float32),
                "g": np.array([self.g_i[i]], dtype=np.float32),
                "avail_cpu_frac": avail_cpu_frac,
                "avail_bw_frac": avail_bw_frac
            }
        obs["server"] = {
            "req_offload": np.zeros(self.N, dtype=np.int8),
            "req_beta": np.zeros(self.N, dtype=np.float32),
            "req_theta": np.zeros(self.N, dtype=np.float32),
            "s": self.s.astype(np.float32),
            "c": self.c.astype(np.float32),
            "D": self.D.astype(np.float32),
            "g": self.g_i.astype(np.float32),
            "avail_cpu_frac": avail_cpu_frac,
            "avail_bw_frac": avail_bw_frac
        }
        return obs

    def obs_space(self, agent_id: str):
        return self.observation_spaces[agent_id]

    def act_space(self, agent_id: str):
        return self.action_spaces[agent_id]

    # rate formula R = theta * W * log2(1 + p*g/(theta*W*N0))
    def _rate(self, p: float, g: float, theta: float) -> float:
        bw = max(theta * self.W, 1e-9)
        snr = (p * g) / (bw * self.N0)
        return bw * np.log2(1.0 + snr + 1e-12)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        # parse UE proposals
        req_offload = np.zeros(self.N, dtype=np.int8)
        req_beta = np.zeros(self.N, dtype=np.float32)
        req_theta = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            a = action.get(f"ue_{i}", None)
            if a is None:
                req_offload[i] = 0
                req_beta[i] = 0.0
                req_theta[i] = 0.0
            else:
                req_offload[i] = int(a.get("offload", 0))
                req_beta[i] = float(np.clip(np.array(a.get("beta", [0.0]))[0], 0.0, 1.0))
                req_theta[i] = float(np.clip(np.array(a.get("theta", [0.0]))[0], 0.0, 1.0))

        server_act = action.get("server", None)
        if server_act is None:
            # accept all requested offloads and equal split
            accept = req_offload.copy()
            accepted = np.where(accept == 1)[0]
            if len(accepted) > 0:
                beta_alloc = np.zeros(self.N, dtype=np.float32)
                theta_alloc = np.zeros(self.N, dtype=np.float32)
                beta_alloc[accepted] = 1.0 / len(accepted)
                theta_alloc[accepted] = 1.0 / len(accepted)
            else:
                beta_alloc = np.zeros(self.N, dtype=np.float32)
                theta_alloc = np.zeros(self.N, dtype=np.float32)
        else:
            accept = np.asarray(server_act.get("accept", np.zeros(self.N)), dtype=np.int8)
            beta_alloc = np.asarray(server_act.get("beta_alloc", np.zeros(self.N)), dtype=np.float32)
            theta_alloc = np.asarray(server_act.get("theta_alloc", np.zeros(self.N)), dtype=np.float32)
            accepted = np.where(accept == 1)[0]
            if len(accepted) > 0:
                # normalize positive allocations among accepted only
                pos = np.maximum(beta_alloc[accepted], 1e-6)
                beta_alloc[accepted] = pos / np.sum(pos)
                pos2 = np.maximum(theta_alloc[accepted], 1e-6)
                theta_alloc[accepted] = pos2 / np.sum(pos2)
            else:
                beta_alloc[:] = 0.0
                theta_alloc[:] = 0.0

        # compute delays/energies
        delays = np.zeros(self.N, dtype=np.float32)
        energies = np.zeros(self.N, dtype=np.float32)
        qos_vals = np.zeros(self.N, dtype=np.float32)

        tx_times = np.zeros(self.N, dtype=np.float32)
        exe_times = np.zeros(self.N, dtype=np.float32)
        tx_energy = np.zeros(self.N, dtype=np.float32)
        exe_energy = np.zeros(self.N, dtype=np.float32)
        idle_energy = np.zeros(self.N, dtype=np.float32)

        # offloaded tasks compute tx and exe times
        for i in range(self.N):
            if req_offload[i] == 1 and accept[i] == 1:
                theta = float(theta_alloc[i])
                beta = float(beta_alloc[i])
                R = self._rate(self.p_tx[i], self.g_i[i], theta)
                R = max(R, 1e-9)
                tx_t = self.s[i] / R
                tx_e = self.p_tx[i] * tx_t
                exe_t = self.c[i] / (max(beta, 1e-9) * self.f_mec)
                exe_e = self.P_mec * beta * exe_t
                idl_e = self.p_idle[i] * exe_t
                tx_times[i] = tx_t
                exe_times[i] = exe_t
                tx_energy[i] = tx_e
                exe_energy[i] = exe_e
                idle_energy[i] = idl_e

        # EDF schedule for accepted offloaded tasks (execution serialized)
        offloaded_indices = [i for i in range(self.N) if req_offload[i] == 1 and accept[i] == 1]
        schedule_order = sorted(offloaded_indices, key=lambda idx: self.D[idx])
        exec_queue_time = 0.0
        for i in schedule_order:
            transmission_done = tx_times[i]
            start_exec = max(transmission_done, exec_queue_time)
            finish_exec = start_exec + exe_times[i]
            delays[i] = finish_exec
            energies[i] = tx_energy[i] + exe_energy[i] + idle_energy[i]
            exec_queue_time = finish_exec

        # non-offloaded or rejected run locally
        for i in range(self.N):
            if not (req_offload[i] == 1 and accept[i] == 1):
                d_local = float(self.c[i] / self.f_loc[i])
                delta_loc = 1e-27 * (self.f_loc[i] ** 2)
                e_local = float(self.c[i] * delta_loc)
                delays[i] = d_local
                energies[i] = e_local

        for i in range(self.N):
            qos_vals[i] = qos_function(delays[i], self.D[i])

        E_total = float(np.sum(energies))
        QoS_total = float(np.sum(qos_vals))

        # rewards
        alpha = 1.0
        global_reward = -E_total + alpha * QoS_total
        rewards = {f"ue_{i}": float(-energies[i] + 0.2 * qos_vals[i]) for i in range(self.N)}
        rewards["server"] = float(global_reward)

        # next obs
        self._sample_slot_tasks()
        obs = self._get_obs()

        # done flags
        self.step_count += 1
        done_bool = self.step_count >= self.max_steps
        dones = {agent: done_bool for agent in list(rewards.keys())}
        dones["__all__"] = done_bool

        info = {agent: {} for agent in rewards.keys()}
        info["server"]["E_total"] = E_total
        info["server"]["QoS_total"] = QoS_total
        info["server"]["delays"] = delays
        info["server"]["energies"] = energies
        info["server"]["qos_vals"] = qos_vals

        # logging
        self.episode_metrics["per_step_energy"].append(E_total)
        self.episode_metrics["per_step_qos"].append(QoS_total)
        for i in range(self.N):
            self.episode_metrics["per_ue_energy"][i].append(float(energies[i]))
            self.episode_metrics["per_ue_qos"][i].append(float(qos_vals[i]))
        self.episode_metrics["schedule"].append((self.step_count - 1, offloaded_indices, list(exe_times[offloaded_indices])))

        return obs, rewards, dones, info
