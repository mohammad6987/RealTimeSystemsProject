from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Phase2Config:
    """
    Configuration for hierarchical phase-2 experiments.

    Hierarchy:
    - One coordinator agent allocates server CPU shares to clusters.
    - One UE agent per user decides offloading and bandwidth request.
    - One cluster scheduler per cluster handles intra-cluster priority + forwarding.
    """

    # Fixed by project statement for phase 2.
    n_users: int = 100
    n_clusters: int = 4
    aggregate_agents: bool = False

    # Episode/training controls.
    max_steps: int = 200
    seed: int = 0

    # Task generation ranges.
    data_mbit_low: float = 16.0
    data_mbit_high: float = 80.0
    cycles_g_low: float = 1.5
    cycles_g_high: float = 3.5
    deadline_low: float = 2.0
    deadline_high: float = 4.0

    # UE feature ranges.
    dist_low: float = 10.0
    dist_high: float = 50.0
    f_loc_low: float = 0.5e9
    f_loc_high: float = 1.5e9

    # Edge server and communication constants.
    f_mec: float = 6e9
    p_mec: float = 5.0
    bandwidth_hz: float = 20e6
    noise_psd: float = 4e-21
    tx_power_w: float = 0.23
    kappa: float = 1e-27

    # Reward weights.
    w_qos: float = 2.0
    w_energy: float = 5e-2

    # Spectrum safety floor to prevent near-zero-rate starvation.
    theta_min: float = 0.01

    # Coordinator safety floor to prevent cluster starvation.
    phi_min: float = 0.05

    # Fixed upper bound for users per cluster action/obs padding.
    # For K in [4..10], max is ceil(100/4)=25.
    max_users_per_cluster: int = 25

    # Number of Lloyd iterations for lightweight numpy k-means.
    kmeans_iters: int = 25
