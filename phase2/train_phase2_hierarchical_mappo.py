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
    from ray.rllib.algorithms.mappo import MAPPOConfig as AlgoConfig
except Exception:
    from ray.rllib.algorithms.ppo import PPOConfig as AlgoConfig

from phase2.phase2_config import Phase2Config
from phase2.phase2_env import MECPhase2HierarchicalEnv
import phase2.phase2_coordinator_network  # noqa: F401
import phase1.ue_policy_network  # noqa: F401
import phase1.server_scheduler_network  # noqa: F401


def parse_args() -> argparse.Namespace:
    """CLI for running phase-2 cluster sweep experiments."""
    p = argparse.ArgumentParser(description="Train phase-2 hierarchical MAPPO for K in {4..10}, N=100")
    p.add_argument("--cluster-values", nargs="+", type=int, default=[4, 5, 6, 7, 8, 9, 10])
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--n-users", type=int, default=100)
    p.add_argument("--out-dir", type=str, default="plots/phase2")
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(x: List[float], w: int = 5) -> np.ndarray:
    """Simple moving average for smoother learning curves in plots."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.size < w:
        return arr
    kernel = np.ones(w, dtype=np.float32) / float(w)
    y = np.convolve(arr, kernel, mode="valid")
    pad = np.full(w - 1, y[0], dtype=np.float32)
    return np.concatenate([pad, y], axis=0)


def build_algo(env_cfg: Dict[str, Any]):
    """Construct one RLlib algorithm with UE + per-cluster scheduler + global coordinator policies."""
    config = (
        AlgoConfig()
        .environment(env="mec_phase2_env", env_config=env_cfg)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=0, rollout_fragment_length=env_cfg["max_steps"])
        .training(
            lr=3e-4,
            train_batch_size=8000,
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"},
        )
        .multi_agent(
            policies={
                "coordinator_policy": (None, None, None, {"model": {"custom_model": "coordinator_network"}}),
                "ue_policy": (None, None, None, {"model": {"custom_model": "ue_policy_network"}}),
                "cluster_sched_policy": (None, None, None, {"model": {"custom_model": "server_scheduler_network"}}),
            },
            policy_mapping_fn=lambda aid, *args, **kwargs: (
                "coordinator_policy"
                if aid == "coordinator"
                else ("ue_policy" if aid.startswith("ue_") else "cluster_sched_policy")
            ),
        )
    )
    return config.build_algo()


def evaluate_episode(algo, env_cfg: Dict[str, Any], eval_seed: int) -> Dict[str, Any]:
    """Run deterministic evaluation episode and aggregate phase-2 metrics."""
    env = MECPhase2HierarchicalEnv({**env_cfg, "seed": eval_seed})
    obs, _ = env.reset()

    done = False
    while not done:
        actions = {}
        for aid, agent_obs in obs.items():
            if aid == "coordinator":
                policy_id = "coordinator_policy"
            elif aid.startswith("ue_"):
                policy_id = "ue_policy"
            else:
                policy_id = "cluster_sched_policy"

            action = algo.compute_single_action(agent_obs, policy_id=policy_id, explore=False)
            if isinstance(action, tuple):
                action = action[0]
            actions[aid] = action

        obs, _, terms, truncs, _ = env.step(actions)
        done = bool(terms["__all__"] or truncs["__all__"])

    return {
        "sum_qos_episode": float(np.sum(env.logs["sum_qos"])),
        "sum_energy_episode": float(np.sum(env.logs["sum_energy"])),
        "cluster_qos_episode": [float(np.sum(c)) for c in env.logs["cluster_qos"]],
        "cluster_energy_episode": [float(np.sum(c)) for c in env.logs["cluster_energy"]],
        "schedule": env.logs["schedule"],
    }


def evaluate_mean(algo, env_cfg: Dict[str, Any], base_seed: int, n_episodes: int = 5) -> Dict[str, Any]:
    """Average evaluation over multiple episodes for lower-variance curves."""
    stats = [evaluate_episode(algo, env_cfg, eval_seed=base_seed + 1000 + i) for i in range(n_episodes)]

    sum_q = float(np.mean([s["sum_qos_episode"] for s in stats]))
    sum_e = float(np.mean([s["sum_energy_episode"] for s in stats]))

    k = len(stats[0]["cluster_qos_episode"])
    c_q = [float(np.mean([s["cluster_qos_episode"][i] for s in stats])) for i in range(k)]
    c_e = [float(np.mean([s["cluster_energy_episode"][i] for s in stats])) for i in range(k)]

    return {
        "sum_qos_episode": sum_q,
        "sum_energy_episode": sum_e,
        "cluster_qos_episode": c_q,
        "cluster_energy_episode": c_e,
        "schedule": stats[-1]["schedule"],
    }


def plot_total_curve(values: List[float], title: str, y_label: str, out_path: str) -> None:
    """Plot raw and smoothed total metric over learning."""
    x = np.arange(1, len(values) + 1)
    y = np.asarray(values, dtype=np.float32)
    ys = moving_average(values, w=5)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o", linewidth=1.5, alpha=0.45, label="raw")
    plt.plot(x, ys, linewidth=2.2, label="moving avg (w=5)")
    plt.title(title)
    plt.xlabel("Training Iteration")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_cluster(curves: List[List[float]], title: str, y_label: str, out_path: str) -> None:
    """Plot one curve per cluster to compare cluster-level behavior."""
    x = np.arange(1, len(curves[0]) + 1)
    plt.figure(figsize=(10, 5))
    for i, c in enumerate(curves):
        plt.plot(x, c, label=f"Cluster {i}")
    plt.title(title)
    plt.xlabel("Training Iteration")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_schedule_view(schedule_logs: List[Dict[str, Any]], out_path: str) -> None:
    """Show a compact Gantt view for offloaded tasks in first non-empty slots."""
    panels: List[Dict[str, Any]] = []
    for s in schedule_logs:
        slot = int(s["slot"])
        for c in s["clusters"]:
            if c["order"]:
                panels.append({"slot": slot, "cluster": c["cluster"], "order": c["order"], "start": c["start"], "finish": c["finish"]})
            if len(panels) >= 9:
                break
        if len(panels) >= 9:
            break

    if not panels:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, "No offloaded tasks in evaluation schedule.", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.5 * rows), squeeze=False)

    for idx, p in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        for rank, u in enumerate(p["order"]):
            st = p["start"][int(u)]
            ft = p["finish"][int(u)]
            ax.barh(y=rank, width=ft - st, left=st, height=0.7)
        ax.set_title(f"slot={p['slot']} cluster={p['cluster']}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("scheduled order")
        ax.grid(axis="x", alpha=0.2)

    for idx in range(len(panels), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("Phase-2 Edge Scheduling View", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_for_k(k: int, args: argparse.Namespace) -> None:
    """Train one full phase-2 experiment for a chosen cluster count K."""
    out_dir = os.path.join(args.out_dir, f"K_{k}")
    ensure_dir(out_dir)

    env_cfg = asdict(
        Phase2Config(
            n_users=args.n_users,
            n_clusters=k,
            max_steps=args.max_steps,
            seed=100 + k,
            max_users_per_cluster=int(np.ceil(args.n_users / k)),
        )
    )

    algo = build_algo(env_cfg)

    total_qos_curve: List[float] = []
    total_energy_curve: List[float] = []
    cluster_qos_curves: List[List[float]] = [[] for _ in range(k)]
    cluster_energy_curves: List[List[float]] = [[] for _ in range(k)]
    final_schedule: List[Dict[str, Any]] = []

    best_score = -1e30
    best_dir = os.path.join(out_dir, "best")
    ensure_dir(best_dir)

    for it in range(1, args.iterations + 1):
        algo.train()
        stats = evaluate_mean(algo, env_cfg, base_seed=10000 + k * 100 + it, n_episodes=5)

        total_qos_curve.append(stats["sum_qos_episode"])
        total_energy_curve.append(stats["sum_energy_episode"])

        for c in range(k):
            cluster_qos_curves[c].append(stats["cluster_qos_episode"][c])
            cluster_energy_curves[c].append(stats["cluster_energy_episode"][c])

        final_schedule = stats["schedule"]

        score = stats["sum_qos_episode"] - 1e-3 * stats["sum_energy_episode"]
        if score > best_score:
            best_score = score
            algo.save(best_dir)

        print(
            f"K={k} iter={it}/{args.iterations} "
            f"sum_qos={stats['sum_qos_episode']:.3f} "
            f"sum_energy={stats['sum_energy_episode']:.3e}"
        )

    plot_total_curve(
        total_qos_curve,
        title=f"Phase-2 K={k}: Total QoS over Learning",
        y_label="Episode Mean Total QoS",
        out_path=os.path.join(out_dir, "sum_qos_over_learning.png"),
    )
    plot_total_curve(
        total_energy_curve,
        title=f"Phase-2 K={k}: Total Energy over Learning",
        y_label="Episode Mean Total Energy (J)",
        out_path=os.path.join(out_dir, "sum_energy_over_learning.png"),
    )
    plot_per_cluster(
        cluster_qos_curves,
        title=f"Phase-2 K={k}: Per-Cluster QoS over Learning",
        y_label="Episode Mean Cluster QoS",
        out_path=os.path.join(out_dir, "per_cluster_qos_over_learning.png"),
    )
    plot_per_cluster(
        cluster_energy_curves,
        title=f"Phase-2 K={k}: Per-Cluster Energy over Learning",
        y_label="Episode Mean Cluster Energy (J)",
        out_path=os.path.join(out_dir, "per_cluster_energy_over_learning.png"),
    )
    plot_schedule_view(final_schedule, out_path=os.path.join(out_dir, "edge_schedule_view.png"))

    algo.save(out_dir)
    algo.stop()


def main() -> None:
    """Entry point for phase-2 training sweep."""
    args = parse_args()
    ensure_dir(args.out_dir)

    register_env("mec_phase2_env", lambda cfg: MECPhase2HierarchicalEnv(cfg))

    ray.init(ignore_reinit_error=True)
    try:
        for k in args.cluster_values:
            run_for_k(k, args)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
