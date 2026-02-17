from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.tune.registry import register_env

try:
    # Use MAPPOConfig when available in the installed Ray version.
    from ray.rllib.algorithms.mappo import MAPPOConfig as AlgoConfig
except Exception:
    # Fallback for Ray versions that expose only PPOConfig.
    from ray.rllib.algorithms.ppo import PPOConfig as AlgoConfig

from phase1.mec_phase1_config import Phase1Config
from phase1.mec_phase1_env import MECPhase1MAPPOEnv
import phase1.ue_policy_network  # noqa: F401
import phase1.server_scheduler_network  # noqa: F401


def parse_args() -> argparse.Namespace:
    """Read command-line arguments that control the experiment sweep."""
    parser = argparse.ArgumentParser(description="Train phase-1 MAPPO-style system for N in {2..7}")
    parser.add_argument("--n-values", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="plots/phase1")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    """Create output directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def build_algo(env_cfg: Dict[str, Any]):
    """
    Build one RLlib algorithm instance for the provided environment config.

    Multi-agent setup:
    - all UE agents share one UE policy network,
    - server uses a separate policy network.
    """
    config = (
        AlgoConfig()
        .environment(env="mec_phase1_env", env_config=env_cfg)
        .framework("torch")
        .resources(num_gpus=1)
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=4, rollout_fragment_length=env_cfg["max_steps"])
        .training(
            lr=3e-4,
            train_batch_size=8000,
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
        )
        .multi_agent(
            policies={
                # UE policy uses the custom UE model registered by name.
                "ue_policy": (None, None, None, {"model": {"custom_model": "ue_policy_network"}}),
                # Server policy uses the custom scheduler model.
                "server_policy": (None, None, None, {"model": {"custom_model": "server_scheduler_network"}}),
            },
            # Route each agent-id to the corresponding shared policy.
            policy_mapping_fn=lambda aid, *args, **kwargs: "server_policy" if aid == "server" else "ue_policy",
        )
    )
    return config.build_algo()


def evaluate_episode(algo, env_cfg: Dict[str, Any], eval_seed: int) -> Dict[str, Any]:
    """
    Run one deterministic evaluation episode and collect aggregate metrics.

    We use `explore=False` to reduce policy stochasticity in evaluation plots.
    """
    env = MECPhase1MAPPOEnv({**env_cfg, "seed": eval_seed})
    obs, _ = env.reset()

    done = False
    while not done:
        actions = {}
        for aid, agent_obs in obs.items():
            policy_id = "server_policy" if aid == "server" else "ue_policy"
            action = algo.compute_single_action(agent_obs, policy_id=policy_id, explore=False)
            if isinstance(action, tuple):
                action = action[0]
            actions[aid] = action

        obs, _, terminateds, truncateds, _ = env.step(actions)
        done = bool(terminateds["__all__"] or truncateds["__all__"])

    return {
        "sum_qos_episode": float(np.sum(env.logs["sum_qos"])),
        "sum_energy_episode": float(np.sum(env.logs["sum_energy"])),
        "per_ue_qos_episode": [float(np.sum(x)) for x in env.logs["per_ue_qos"]],
        "per_ue_energy_episode": [float(np.sum(x)) for x in env.logs["per_ue_energy"]],
        "scheduling": env.logs["scheduling"],
    }


def plot_sum_curve(values: List[float], title: str, y_label: str, path: str) -> None:
    """Plot one scalar metric over training iterations and save to disk."""
    x = np.arange(1, len(values) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(x, values, linewidth=2, marker="o", markersize=5)
    # Keep a visible x-span even for a single-point run.
    if len(values) == 1:
        plt.xlim(0.5, 1.5)
    plt.title(title)
    plt.xlabel("Training Iteration")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_per_ue_curves(values_per_ue: List[List[float]], title: str, y_label: str, path: str) -> None:
    """Plot one line per UE to show per-user behavior across training."""
    x = np.arange(1, len(values_per_ue[0]) + 1)
    plt.figure(figsize=(9, 5))
    for i, series in enumerate(values_per_ue):
        plt.plot(x, series, label=f"UE {i}", marker="o", markersize=4)
    if len(values_per_ue[0]) == 1:
        plt.xlim(0.5, 1.5)
    plt.title(title)
    plt.xlabel("Training Iteration")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_schedule(scheduling_logs: List[Dict[str, Any]], n_ue: int, path: str) -> None:
    """
    Render a compact schedule view (Gantt-like) from evaluation logs.

    Each panel corresponds to one slot; bars show per-UE execution windows
    for tasks that were actually offloaded.
    """
    selected = [s for s in scheduling_logs if s["order"]][:9]
    if not selected:
        # Keep a non-empty artifact even when no offloading occurs.
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(
            0.5,
            0.5,
            "No offloaded tasks in this evaluation episode.\nNo server schedule to display.",
            ha="center",
            va="center",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return

    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)

    for idx, slot_data in enumerate(selected):
        ax = axes[idx // cols][idx % cols]
        for ue in slot_data["order"]:
            start = slot_data["start"][int(ue)]
            finish = slot_data["finish"][int(ue)]
            ax.barh(y=int(ue), width=finish - start, left=start, height=0.7)
        ax.set_title(f"Slot {slot_data['slot']}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("UE")
        ax.set_yticks(range(n_ue))
        ax.grid(axis="x", alpha=0.2)

    # Hide unused subplot panels when < 9 slots were selected.
    for idx in range(len(selected), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("Edge Server Learned Scheduling View", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_for_n(n_ue: int, iterations: int, max_steps: int, out_dir: str, base_seed: int) -> None:
    """
    Train and evaluate one experiment setting for a fixed number of UEs.

    Outputs:
    - checkpoint,
    - aggregate QoS/energy curves,
    - per-UE QoS/energy curves,
    - schedule visualization.
    """
    target_dir = os.path.join(out_dir, f"N_{n_ue}")
    ensure_dir(target_dir)

    # Build env config with run-specific scale parameters.
    env_cfg = asdict(
        Phase1Config(
            n_ue=n_ue,
            max_steps=max_steps,
            seed=base_seed,
            noise_psd=4e-21,
            tx_power_w=0.23,
            kappa=1e-27,
        )
    )

    algo = build_algo(env_cfg)

    # Curves indexed by training iteration.
    sum_qos_curve: List[float] = []
    sum_energy_curve: List[float] = []
    per_ue_qos_curves: List[List[float]] = [[] for _ in range(n_ue)]
    per_ue_energy_curves: List[List[float]] = [[] for _ in range(n_ue)]
    final_schedule: List[Dict[str, Any]] = []

    for it in range(1, iterations + 1):
        # One PPO update on collected rollout data.
        algo.train()
        # Deterministic episode for progress logging/plotting.
        stats = evaluate_episode(algo, env_cfg, eval_seed=base_seed + 10000 + it)

        sum_qos_curve.append(stats["sum_qos_episode"])
        sum_energy_curve.append(stats["sum_energy_episode"])
        for i in range(n_ue):
            per_ue_qos_curves[i].append(stats["per_ue_qos_episode"][i])
            per_ue_energy_curves[i].append(stats["per_ue_energy_episode"][i])

        final_schedule = stats["scheduling"]
        print(
            f"N={n_ue} iter={it}/{iterations} "
            f"sum_qos={stats['sum_qos_episode']:.3f} "
            f"sum_energy={stats['sum_energy_episode']:.3e}"
        )

    # Persist all required phase-1 figures.
    plot_sum_curve(
        sum_qos_curve,
        title=f"Phase-1 N={n_ue}: Total QoS over Learning",
        y_label="Episode Total QoS",
        path=os.path.join(target_dir, "sum_qos_over_learning.png"),
    )
    plot_sum_curve(
        sum_energy_curve,
        title=f"Phase-1 N={n_ue}: Total Energy over Learning",
        y_label="Episode Total Energy (J)",
        path=os.path.join(target_dir, "sum_energy_over_learning.png"),
    )
    plot_per_ue_curves(
        per_ue_qos_curves,
        title=f"Phase-1 N={n_ue}: Per-UE QoS over Learning",
        y_label="Episode QoS per UE",
        path=os.path.join(target_dir, "per_ue_qos_over_learning.png"),
    )
    plot_per_ue_curves(
        per_ue_energy_curves,
        title=f"Phase-1 N={n_ue}: Per-UE Energy over Learning",
        y_label="Episode Energy per UE (J)",
        path=os.path.join(target_dir, "per_ue_energy_over_learning.png"),
    )
    plot_schedule(
        final_schedule,
        n_ue=n_ue,
        path=os.path.join(target_dir, "edge_schedule_view.png"),
    )

    # Save final trained state for this N.
    algo.save(target_dir)
    algo.stop()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    # Expose env under a stable RLlib name.
    register_env("mec_phase1_env", lambda cfg: MECPhase1MAPPOEnv(cfg))

    ray.init(ignore_reinit_error=True)
    try:
        for n_ue in args.n_values:
            run_for_n(
                n_ue=n_ue,
                iterations=args.iterations,
                max_steps=args.max_steps,
                out_dir=args.out_dir,
                base_seed=42 + n_ue,
            )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
