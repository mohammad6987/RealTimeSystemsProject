from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from phase1.mec_phase1_config import Phase1Config
from phase1.sb3_phase1_env import SB3Phase1JointEnv
from phase1.ue_policy_network import UEPolicySB3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phase-1 MAPPO-style system using SB3 PPO")
    parser.add_argument("--n-values", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--timesteps-per-iter", type=int, default=2000)
    parser.add_argument("--out-dir", type=str, default="plots/phase1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_sum_curve(values: List[float], title: str, y_label: str, path: str) -> None:
    x = np.arange(1, len(values) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(x, values, linewidth=2, marker="o", markersize=5)
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
    selected = [s for s in scheduling_logs if s["order"]][:9]
    if not selected:
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

    for idx in range(len(selected), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("Edge Server Learned Scheduling View", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_episode(model: PPO, env_cfg: Dict[str, Any], eval_seed: int) -> Dict[str, Any]:
    eval_env = SB3Phase1JointEnv({**env_cfg, "seed": eval_seed})
    obs, _ = eval_env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        done = bool(terminated or truncated)

    return {
        "sum_qos_episode": float(np.sum(eval_env.logs["sum_qos"])),
        "sum_energy_episode": float(np.sum(eval_env.logs["sum_energy"])),
        "per_ue_qos_episode": [float(np.sum(x)) for x in eval_env.logs["per_ue_qos"]],
        "per_ue_energy_episode": [float(np.sum(x)) for x in eval_env.logs["per_ue_energy"]],
        "scheduling": eval_env.logs["scheduling"],
    }


def run_for_n(
    n_ue: int,
    iterations: int,
    max_steps: int,
    timesteps_per_iter: int,
    out_dir: str,
    base_seed: int,
) -> None:
    target_dir = os.path.join(out_dir, f"N_{n_ue}")
    ensure_dir(target_dir)

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

    def make_env():
        return SB3Phase1JointEnv(env_cfg)

    vec_env = DummyVecEnv([make_env])

    if UEPolicySB3 is None:
        raise RuntimeError("Stable-Baselines3 is required for SB3 training. Install via `pip install stable-baselines3`.")

    model = PPO(
        policy=UEPolicySB3,
        env=vec_env,
        learning_rate=3e-5,
        batch_size=4096,
        n_steps=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"activation_fn": nn.ReLU, "net_arch": [256, 256]},
        seed=base_seed,
        verbose=0,
    )

    sum_qos_curve: List[float] = []
    sum_energy_curve: List[float] = []
    per_ue_qos_curves: List[List[float]] = [[] for _ in range(n_ue)]
    per_ue_energy_curves: List[List[float]] = [[] for _ in range(n_ue)]
    final_schedule: List[Dict[str, Any]] = []

    for it in range(1, iterations + 1):
        model.learn(total_timesteps=timesteps_per_iter, reset_num_timesteps=False)
        stats = evaluate_episode(model, env_cfg, eval_seed=base_seed + 10000 + it)

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

    plot_sum_curve(
        sum_qos_curve,
        title=f"Phase-1 N={n_ue}: Total QoS over Learning (SB3)",
        y_label="Episode Total QoS",
        path=os.path.join(target_dir, "sum_qos_over_learning.png"),
    )
    plot_sum_curve(
        sum_energy_curve,
        title=f"Phase-1 N={n_ue}: Total Energy over Learning (SB3)",
        y_label="Episode Total Energy (J)",
        path=os.path.join(target_dir, "sum_energy_over_learning.png"),
    )
    plot_per_ue_curves(
        per_ue_qos_curves,
        title=f"Phase-1 N={n_ue}: Per-UE QoS over Learning (SB3)",
        y_label="Episode QoS per UE",
        path=os.path.join(target_dir, "per_ue_qos_over_learning.png"),
    )
    plot_per_ue_curves(
        per_ue_energy_curves,
        title=f"Phase-1 N={n_ue}: Per-UE Energy over Learning (SB3)",
        y_label="Episode Energy per UE (J)",
        path=os.path.join(target_dir, "per_ue_energy_over_learning.png"),
    )
    plot_schedule(final_schedule, n_ue=n_ue, path=os.path.join(target_dir, "edge_schedule_view.png"))

    model.save(os.path.join(target_dir, "sb3_ppo_model"))


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for n_ue in args.n_values:
        run_for_n(
            n_ue=n_ue,
            iterations=args.iterations,
            max_steps=args.max_steps,
            timesteps_per_iter=args.timesteps_per_iter,
            out_dir=args.out_dir,
            base_seed=args.seed + n_ue,
        )


if __name__ == "__main__":
    main()
