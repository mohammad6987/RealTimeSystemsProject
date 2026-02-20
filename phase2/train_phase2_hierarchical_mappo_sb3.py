from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from phase2.phase2_config import Phase2Config
from phase2.sb3_phase2_env import SB3Phase2JointEnv
from phase1.ue_policy_network import UEPolicySB3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train phase-2 hierarchical MAPPO using SB3 PPO")
    p.add_argument("--cluster-values", nargs="+", type=int, default=[4, 5, 6, 7, 8, 9, 10])
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--n-users", type=int, default=100)
    p.add_argument("--timesteps-per-iter", type=int, default=3000)
    p.add_argument("--out-dir", type=str, default="plots/phase2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--schedule-every", type=int, default=10)
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(x: List[float], w: int = 5) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.size < w:
        return arr
    kernel = np.ones(w, dtype=np.float32) / float(w)
    y = np.convolve(arr, kernel, mode="valid")
    pad = np.full(w - 1, y[0], dtype=np.float32)
    return np.concatenate([pad, y], axis=0)


def plot_total_curve(values: List[float], title: str, y_label: str, out_path: str) -> None:
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
    if not schedule_logs:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, "No offloaded tasks in evaluation schedule.", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return
    has_tasks = any(c["order"] for s in schedule_logs for c in s["clusters"])
    if not has_tasks:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, "No offloaded tasks in evaluation schedule.", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    max_cluster = -1
    for s in schedule_logs:
        for c in s["clusters"]:
            max_cluster = max(max_cluster, int(c["cluster"]))
    k = max_cluster + 1
    if k <= 0:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, "No cluster schedules found.", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    total_slots = len(schedule_logs)
    total_rows = max(1, total_slots * k)
    fig_height = min(120.0, max(6.0, 0.08 * total_rows))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    cmap = plt.get_cmap("tab10", k)
    for s in schedule_logs:
        slot = int(s["slot"])
        for c in s["clusters"]:
            cluster = int(c["cluster"])
            color = cmap(cluster % cmap.N)
            for u in c["order"]:
                st = c["start"][int(u)]
                ft = c["finish"][int(u)]
                y = slot * k + cluster
                ax.barh(y=y, width=ft - st, left=st, height=0.8, color=color)

    ax.set_title("Phase-2 Edge Scheduling View (All Slots)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("slot / cluster")
    slot_ticks = [s * k + (k - 1) / 2.0 for s in range(total_slots)]
    ax.set_yticks(slot_ticks)
    ax.set_yticklabels([f"slot {s}" for s in range(total_slots)], fontsize=7)
    ax.grid(axis="x", alpha=0.2)

    handles = [mpatches.Patch(color=cmap(i), label=f"cluster {i}") for i in range(k)]
    ax.legend(handles=handles, ncol=min(k, 5), fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_episode(model: PPO, env_cfg: Dict[str, Any], eval_seed: int) -> Dict[str, Any]:
    eval_env = SB3Phase2JointEnv({**env_cfg, "seed": eval_seed})
    obs, _ = eval_env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        done = bool(terminated or truncated)

    return {
        "sum_qos_episode": float(np.sum(eval_env.logs["sum_qos"])),
        "sum_energy_episode": float(np.sum(eval_env.logs["sum_energy"])),
        "cluster_qos_episode": [float(np.sum(c)) for c in eval_env.logs["cluster_qos"]],
        "cluster_energy_episode": [float(np.sum(c)) for c in eval_env.logs["cluster_energy"]],
        "schedule": eval_env.logs["schedule"],
    }


def evaluate_mean(model: PPO, env_cfg: Dict[str, Any], base_seed: int, n_episodes: int = 5) -> Dict[str, Any]:
    stats = [evaluate_episode(model, env_cfg, eval_seed=base_seed + 1000 + i) for i in range(n_episodes)]

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


def run_for_k(k: int, args: argparse.Namespace) -> None:
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

    def make_env():
        return SB3Phase2JointEnv(env_cfg)

    vec_env = DummyVecEnv([make_env])

    if UEPolicySB3 is None:
        raise RuntimeError("Stable-Baselines3 is required for SB3 training. Install via `pip install stable-baselines3`.")

    model = PPO(
        policy=UEPolicySB3,
        env=vec_env,
        learning_rate=3e-4,
        batch_size=1024,
        n_steps=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"activation_fn": nn.ReLU, "net_arch": [256, 256]},
        seed=args.seed,
        verbose=0,
    )

    total_qos_curve: List[float] = []
    total_energy_curve: List[float] = []
    cluster_qos_curves: List[List[float]] = [[] for _ in range(k)]
    cluster_energy_curves: List[List[float]] = [[] for _ in range(k)]
    final_schedule: List[Dict[str, Any]] = []

    best_score = -1e30
    best_dir = os.path.join(out_dir, "best")
    ensure_dir(best_dir)

    for it in range(1, args.iterations + 1):
        model.learn(total_timesteps=args.timesteps_per_iter, reset_num_timesteps=False)
        stats = evaluate_mean(model, env_cfg, base_seed=10000 + k * 100 + it, n_episodes=5)

        total_qos_curve.append(stats["sum_qos_episode"])
        total_energy_curve.append(stats["sum_energy_episode"])

        for c in range(k):
            cluster_qos_curves[c].append(stats["cluster_qos_episode"][c])
            cluster_energy_curves[c].append(stats["cluster_energy_episode"][c])

        final_schedule = stats["schedule"]

        if args.schedule_every > 0 and (it % args.schedule_every == 0):
            plot_schedule_view(
                final_schedule,
                out_path=os.path.join(out_dir, f"edge_schedule_view_iter_{it}.png"),
            )

        score = stats["sum_qos_episode"] - 1e-3 * stats["sum_energy_episode"]
        if score > best_score:
            best_score = score
            model.save(os.path.join(best_dir, "sb3_ppo_model"))

        print(
            f"K={k} iter={it}/{args.iterations} "
            f"sum_qos={stats['sum_qos_episode']:.3f} "
            f"sum_energy={stats['sum_energy_episode']:.3e}"
        )

    plot_total_curve(
        total_qos_curve,
        title=f"Phase-2 K={k}: Total QoS over Learning (SB3)",
        y_label="Episode Mean Total QoS",
        out_path=os.path.join(out_dir, "sum_qos_over_learning.png"),
    )
    plot_total_curve(
        total_energy_curve,
        title=f"Phase-2 K={k}: Total Energy over Learning (SB3)",
        y_label="Episode Mean Total Energy (J)",
        out_path=os.path.join(out_dir, "sum_energy_over_learning.png"),
    )
    plot_per_cluster(
        cluster_qos_curves,
        title=f"Phase-2 K={k}: Per-Cluster QoS over Learning (SB3)",
        y_label="Episode Mean Cluster QoS",
        out_path=os.path.join(out_dir, "per_cluster_qos_over_learning.png"),
    )
    plot_per_cluster(
        cluster_energy_curves,
        title=f"Phase-2 K={k}: Per-Cluster Energy over Learning (SB3)",
        y_label="Episode Mean Cluster Energy (J)",
        out_path=os.path.join(out_dir, "per_cluster_energy_over_learning.png"),
    )
    plot_schedule_view(final_schedule, out_path=os.path.join(out_dir, "edge_schedule_view.png"))

    model.save(os.path.join(out_dir, "sb3_ppo_model"))


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for k in args.cluster_values:
        run_for_k(k, args)


if __name__ == "__main__":
    main()
