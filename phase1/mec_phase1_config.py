from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Phase1Config:
    seed: int = 42
    # Number of UE agents in one experiment run.
    n_ue: int = 5
    # Number of time slots per episode.
    max_steps: int = 200
    # generated Tasks size
    data_mbit_low: float = 16.0
    data_mbit_high: float = 80.0
    # CPU workload in Giga cycles
    cycles_g_low: float = 1.5
    cycles_g_high: float = 3.5
    # Task deadline in seconds
    deadline_low: float = 1.0
    deadline_high: float = 2.0

    dist_low: float = 10.0
    dist_high: float = 50.0
    # UE local CPU frequency
    f_loc_low: float = 0.5e9
    f_loc_high: float = 1.5e9

    # Edge server compute frequency.
    f_mec: float = 6e9
    # Edge server power draw in 
    p_mec: float = 5.0
    # Total uplink spectrum shared among offloaded UEs.
    bandwidth_hz: float = 10e6

    # Channel noise PSD (W/Hz).
    noise_psd: float = 4e-21
    # UE transmit power in Watts when offloading.
    tx_power_w: float = 0.08
    # Dynamic power coefficient for local CPU energy model.
    kappa: float = 1e-27

    # reward = w_qos * sum_qos - w_energy * total_energy
    w_qos: float = 1.0
    w_energy: float = 1e-6
