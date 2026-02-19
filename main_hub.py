from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


def run_command(cmd: List[str]) -> int:
    """Execute a child command and stream output directly to terminal."""
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def main() -> None:
    """
    One entry-point for both project phases.

    Examples:
      python main_hub.py phase1 --n-values 2 3 --iterations 20 --max-steps 30
      python main_hub.py phase2 --cluster-values 4 5 --n-users 100 --iterations 20 --max-steps 30
    """
    parser = argparse.ArgumentParser(description="Main hub for phase-1 and phase-2 training")
    sub = parser.add_subparsers(dest="phase", required=True)

    p1 = sub.add_parser("phase1", help="Run phase-1 training")
    p1.add_argument("--n-values", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    p1.add_argument("--iterations", type=int, default=120)
    p1.add_argument("--max-steps", type=int, default=200)
    p1.add_argument("--out-dir", type=str, default="plots/phase1")
    p1.add_argument("--backend", choices=["rllib", "sb3"], default="rllib")
    p1.add_argument("--seed", type=int, default=42)

    p2 = sub.add_parser("phase2", help="Run phase-2 hierarchical training")
    p2.add_argument("--cluster-values", nargs="+", type=int, default=[4, 5, 6, 7, 8, 9, 10])
    p2.add_argument("--iterations", type=int, default=120)
    p2.add_argument("--max-steps", type=int, default=200)
    p2.add_argument("--n-users", type=int, default=100)
    p2.add_argument("--out-dir", type=str, default="plots/phase2")
    p2.add_argument("--backend", choices=["rllib", "sb3"], default="rllib")
    p2.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.phase == "phase1":
        module = "phase1.train_phase1_mappo" if args.backend == "rllib" else "phase1.train_phase1_mappo_sb3"
        cmd = [
            sys.executable,
            "-m",
            module,
            "--n-values",
            *[str(v) for v in args.n_values],
            "--iterations",
            str(args.iterations),
            "--max-steps",
            str(args.max_steps),
            "--out-dir",
            args.out_dir,
        ]
        if args.backend == "sb3":
            cmd += ["--seed", str(args.seed)]
    else:
        module = (
            "phase2.train_phase2_hierarchical_mappo"
            if args.backend == "rllib"
            else "phase2.train_phase2_hierarchical_mappo_sb3"
        )
        cmd = [
            sys.executable,
            "-m",
            module,
            "--cluster-values",
            *[str(v) for v in args.cluster_values],
            "--iterations",
            str(args.iterations),
            "--max-steps",
            str(args.max_steps),
            "--n-users",
            str(args.n_users),
            "--out-dir",
            args.out_dir,
        ]
        if args.backend == "sb3":
            cmd += ["--seed", str(args.seed)]

    code = run_command(cmd)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
