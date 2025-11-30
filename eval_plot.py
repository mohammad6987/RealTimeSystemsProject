# eval_and_plot.py
"""Evaluate a trained checkpoint and produce required plots."""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer
from mec_env import MECEnv

def evaluate(checkpoint_path, N=5, episodes=20, seed=0):
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    env = MECEnv({"N": N, "seed": seed, "max_steps": 200})
    config = {"env": None, "framework": "torch"}  
    trainer = PPOTrainer(config=config, env=env)
    trainer.restore(checkpoint_path)

    total_energy = []
    total_qos = []
    per_ue_energy = [[] for _ in range(N)]
    per_ue_qos = [[] for _ in range(N)]

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action_dict = {}
            for i in range(N):
                agent_id = f"ue_{i}"
                policy_id = "ue_policy"
                action = trainer.compute_action(obs[agent_id], policy_id=policy_id)
                # RLlib returns concatenated vector from custom model; but trainer.compute_action will
                # return the action in the action-space shape if policy/action-distribution configured.
                action_dict[agent_id] = action
            # server
            server_action = trainer.compute_action(obs["server"], policy_id="server_policy")
            action_dict["server"] = server_action

            obs, rewards, dones, info = env.step(action_dict)
            # info stored under server
            E = info["server"]["E_total"]
            Q = info["server"]["QoS_total"]
            total_energy.append(E)
            total_qos.append(Q)
            for i in range(N):
                per_ue_energy[i].append(env.episode_metrics["per_ue_energy"][i][-1])
                per_ue_qos[i].append(env.episode_metrics["per_ue_qos"][i][-1])
            done = dones["__all__"]

    # plots
    t = np.arange(len(total_energy))
    plt.figure(figsize=(8,4)); plt.plot(t, total_energy); plt.title("Total Energy"); plt.xlabel("step"); plt.savefig("eval_total_energy.png")
    plt.figure(figsize=(8,4)); plt.plot(t, total_qos); plt.title("Total QoS"); plt.xlabel("step"); plt.savefig("eval_total_qos.png")

    # per-user plots (averages)
    avg_qos = [np.mean(per_ue_qos[i]) for i in range(N)]
    avg_energy = [np.mean(per_ue_energy[i]) for i in range(N)]
    plt.figure(figsize=(8,4)); plt.bar(range(N), avg_qos); plt.title("Per-UE Avg QoS"); plt.savefig("eval_per_user_qos.png")
    plt.figure(figsize=(8,4)); plt.bar(range(N), avg_energy); plt.title("Per-UE Avg Energy"); plt.savefig("eval_per_user_energy.png")

    # schedule timeline simple scatter
    schedule = env.episode_metrics["schedule"]
    plt.figure(figsize=(8,4))
    for step, accepted, exe_times in schedule[-200:]:
        for idx, t_exec in zip(accepted, exe_times):
            plt.scatter(step, idx, s=max(1, t_exec*1000))
    plt.title("Schedule: accepted tasks (last steps)")
    plt.savefig("eval_schedule.png")
    print("Saved plots: eval_total_energy.png, eval_total_qos.png, eval_per_user_qos.png, eval_per_user_energy.png, eval_schedule.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    evaluate(args.checkpoint, N=args.N, episodes=args.episodes, seed=args.seed)
