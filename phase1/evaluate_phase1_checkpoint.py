from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from phase1.mec_phase1_env import MECPhase1MAPPOEnv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for quick post-training sanity visualization."""
    parser = argparse.ArgumentParser(description="Visualize one evaluation episode from saved logs npy.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path like plots/phase1/N_5")
    return parser.parse_args()


def main() -> None:
    """
    Minimal utility script:
    - checks whether scheduling figure already exists,
    - generates a small observation sanity plot for UE-0.
    """
    args = parse_args()
    schedule_path = os.path.join(args.input_dir, "edge_schedule_view.png")
    if os.path.exists(schedule_path):
        print(f"Scheduling figure already exists: {schedule_path}")
    else:
        print("No scheduling figure found. Train first with train_phase1_mappo.py")

    # Create a fresh env and inspect one sampled observation.
    env = MECPhase1MAPPOEnv()
    obs, _ = env.reset(seed=123)
    ue0 = obs["ue_0"]

    # Bar chart helps verify feature scales after any config changes.
    plt.figure(figsize=(6, 3))
    plt.bar(["data", "cycles", "deadline", "dist", "f_loc"], ue0)
    plt.title("UE-0 Observation Sample")
    plt.tight_layout()
    out = os.path.join(args.input_dir, "sanity_ue0_obs.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
