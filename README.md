# RealTimeSystemsProject

Hierarchical reinforcement-learning implementation for mobile edge computing (MEC), covering:

- **Phase 1**: flat multi-agent control (UE agents + one server scheduler).
- **Phase 2**: hierarchical control (global coordinator + per-cluster controllers).

This project uses **Ray RLlib** with **TorchModelV2 custom models** (shared policy per role).

## Repository Structure

```text
.
├── main_hub.py
├── phase1/
│   ├── mec_phase1_config.py
│   ├── mec_phase1_env.py
│   ├── ue_policy_network.py
│   ├── server_scheduler_network.py
│   ├── train_phase1_mappo.py
│   └── evaluate_phase1_checkpoint.py
├── phase2/
│   ├── phase2_config.py
│   ├── phase2_env.py
│   ├── phase2_coordinator_network.py
│   ├── phase2_cluster_network.py
│   └── train_phase2_hierarchical_mappo.py
├── plots/
│   ├── phase1/
│   └── phase2/
└── Project_20.pdf
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r reqs.txt
```

## Main Hub (Recommended)

Use one command entry-point for both phases.

- Phase 1:

```bash
python main_hub.py phase1 --n-values 2 3 4 5 6 7 --iterations 120 --max-steps 200
```

- Phase 2:

```bash
python main_hub.py phase2 --cluster-values 4 5 6 7 8 9 10 --n-users 100 --iterations 120 --max-steps 200
```

## Phase 1

### Goal

For each time slot and each UE:
- decide local execution vs offloading,
- if offloading, compete for uplink spectrum,
- server policy learns priority-based scheduling order for offloaded tasks.

### Agent Structure

- Agents: `ue_0 ... ue_{N-1}`, `server`
- Policies:
  - `ue_policy` shared by all UE agents
  - `server_policy` for server scheduler
- Reward: cooperative shared reward to all agents
  - `reward = w_qos * total_qos - w_energy * total_energy`

### Observation/Action Spaces

- UE observation (`shape=(5,)`):
  - `[data_mbit, cycles_g, deadline_s, dist_m, f_loc_ghz]`
- UE action (`shape=(2,)`):
  - `[offload_logit, spectrum_request_logit]`

- Server observation (`shape=(N*6,)`, flattened per-UE table):
  - `[data_norm, cycles_norm, deadline_norm, dist_norm, req_offload, req_spectrum_share]`
- Server action (`shape=(N,)`):
  - priority scores per UE

### Environment Dynamics (per slot)

1. UEs emit offload and spectrum-request logits.
2. Offloaded UEs get normalized spectrum shares (softmax).
3. Transmission delay/energy is computed for offloaded tasks.
4. Local delay/energy is computed for non-offloaded tasks.
5. Server orders offloaded tasks by learned priority (arrival-aware serial scheduling).
6. QoS is computed by deadline-based piecewise function.
7. Global reward is computed and broadcast.

### Phase-1 Default Parameters (`phase1/mec_phase1_config.py`)

- Users: `n_ue=5` (overridden by CLI sweep)
- Steps: `max_steps=200`
- Task size: `[16, 80] Mbit`
- Task cycles: `[1.5, 3.5] Gcycles`
- Deadline: `[1.0, 2.0] s`
- Distance: `[10, 50] m`
- UE CPU: `[0.5, 1.5] GHz`
- Server CPU: `6 GHz`
- Server power: `5 W`
- Bandwidth: `50 MHz`
- Tx power: `0.08 W` (trainer may override)
- Noise PSD: `4e-21 W/Hz`
- `kappa=1e-27`
- Reward: `w_qos=1.0`, `w_energy=1e-6`

### Models (Phase 1)

- `phase1/ue_policy_network.py`
  - `UEPolicyNetwork(TorchModelV2, nn.Module)`
  - FC backbone via RLlib `FullyConnectedNetwork`
- `phase1/server_scheduler_network.py`
  - `ServerSchedulerNetwork(TorchModelV2, nn.Module)`
  - FC backbone via RLlib `FullyConnectedNetwork`

### Trainer

- File: `phase1/train_phase1_mappo.py`
- Uses RLlib MAPPO config when available, PPO fallback otherwise.
- Uses old RLlib API stack flags for `TorchModelV2` compatibility.
- Produces one checkpoint per `N` run.

### Outputs (Phase 1)

Per `N` in `plots/phase1/N_<N>/`:
- `sum_qos_over_learning.png`
- `sum_energy_over_learning.png`
- `per_ue_qos_over_learning.png`
- `per_ue_energy_over_learning.png`
- `edge_schedule_view.png`
- `rllib_checkpoint.json`, `algorithm_state.pkl`, `policies/...`

## Phase 2 (Hierarchical)

### Goal

Implement hierarchical resource allocation with total users fixed at 100:
- coordinator allocates server compute shares to clusters,
- cluster controllers manage offloading and local scheduling priorities.

Target cluster counts:
- `K ∈ {4,5,6,7,8,9,10}`

### Hierarchical Agent Structure

- Agents:
  - `coordinator`
  - `cluster_0 ... cluster_{K-1}`
- Policies:
  - `coordinator_policy` (single global)
  - `cluster_policy` (shared across clusters)
- Reward:
  - cooperative shared reward with QoS-energy tradeoff

### Clustering Method

- Implemented inside `phase2/phase2_env.py`
- Lightweight NumPy k-means (`run_kmeans`) on static UE features:
  - normalized distance
  - normalized local CPU frequency
- Empty-cluster recovery is included.

### Observation/Action Spaces

- Coordinator observation (`shape=(K*5,)`): per-cluster aggregates
  - `[mean_load, mean_deadline, mean_dist, mean_f_loc, last_offload_ratio]`
- Coordinator action (`shape=(K,)`): cluster allocation logits
  - converted to normalized cluster CPU shares `phi_k`

- Cluster observation (`shape=(M*6,)`): padded local user table (`M=max_users_per_cluster`)
  - per row: `[valid, data_norm, cycles_norm, deadline_norm, dist_norm, f_loc_norm]`
- Cluster action (`shape=(3*M,)`):
  - `[offload_logits, bandwidth_logits, priority_logits]`

### Environment Dynamics (per slot)

1. Coordinator allocates server compute share per cluster.
2. Each cluster controller decides offloading, bandwidth requests, and priorities for local users.
3. Global spectrum is distributed across all offloaded users.
4. Per-cluster scheduling is applied with cluster-specific effective CPU.
5. QoS and energy are computed globally and per cluster.
6. Shared reward is emitted to all agents.

### Phase-2 Default Parameters (`phase2/phase2_config.py`)

- Users: `n_users=100`
- Clusters: `n_clusters=4` (overridden by CLI sweep)
- Steps: `max_steps=200`
- Task size: `[16, 80] Mbit`
- Task cycles: `[1.5, 3.5] Gcycles`
- Deadline: `[2.0, 4.0] s`
- Distance: `[10, 50] m`
- UE CPU: `[0.5, 1.5] GHz`
- Server CPU: `6 GHz`
- Server power: `5 W`
- Bandwidth: `20 MHz`
- Tx power: `0.23 W`
- Noise PSD: `4e-21 W/Hz`
- `kappa=1e-27`
- Reward: `w_qos=1.0`, `w_energy=1e-7`
- Spectrum floor: `theta_min=0.01`
- KMeans iterations: `25`

### Models (Phase 2)

- `phase2/phase2_coordinator_network.py`
  - `CoordinatorNetwork(TorchModelV2, nn.Module)`
- `phase2/phase2_cluster_network.py`
  - `ClusterControllerNetwork(TorchModelV2, nn.Module)`

Both use RLlib FC actor-critic backbone.

### Trainer

- File: `phase2/train_phase2_hierarchical_mappo.py`
- Evaluates each iteration on multiple episodes (`n_episodes=5`) for lower variance.
- Saves:
  - final checkpoint under `plots/phase2/K_<K>/`
  - best checkpoint under `plots/phase2/K_<K>/best/`

### Outputs (Phase 2)

Per `K` in `plots/phase2/K_<K>/`:
- `sum_qos_over_learning.png`
- `sum_energy_over_learning.png`
- `per_cluster_qos_over_learning.png`
- `per_cluster_energy_over_learning.png`
- `edge_schedule_view.png`
- `rllib_checkpoint.json`, `algorithm_state.pkl`, `policies/...`

## Direct Training Commands

If needed, bypass hub:

```bash
python -m phase1.train_phase1_mappo
python -m phase2.train_phase2_hierarchical_mappo
```

## Practical Notes

- RLlib may print deprecation warnings (`compute_single_action`, logger warnings).
  - Current code remains compatible with your installed stack.
- Ray metrics-exporter warnings (`rpc_code:14`) are typically non-fatal.
- For stable conclusions, prefer longer runs and compare several seeds.
- If late-iteration energy spikes appear in phase 2, tune:
  - `w_energy`,
  - learning rate,
  - train batch size,
  - and evaluate with more episodes per iteration.
