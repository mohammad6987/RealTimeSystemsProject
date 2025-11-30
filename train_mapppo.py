import argparse
import ray
from ray import tune
from ray.tune.registry import register_env
from mec_env import MECEnv
import ue_model 
import server_model  

def policy_mapping_fn(agent_id):
    if agent_id.startswith("ue_"):
        return "ue_policy"
    return "server_policy"

def create_env(env_config):
    return MECEnv(env_config)

def main(args):
    ray.init(ignore_reinit_error=True)
    env_name = "mec_env_v1"
    register_env(env_name, lambda cfg: create_env(cfg))

    # create dummy env to get spaces
    dummy = MECEnv({"N": args.N, "seed": args.seed})

    policies = {
        "ue_policy": (None, dummy.obs_space("ue_0"), dummy.act_space("ue_0"), {"model": {"custom_model": "ue_model"}}),
        "server_policy": (None, dummy.obs_space("server"), dummy.act_space("server"), {"model": {"custom_model": "server_model"}})
    }

    config = {
        "env": env_name,
        "env_config": {"N": args.N, "seed": args.seed, "slot_time": args.slot_time, "max_steps": args.max_steps},
        "num_workers": max(0, args.num_cpus - 1),
        "framework": "torch",
        "log_level": "INFO",
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda agent_id, episode, **kwargs: policy_mapping_fn(agent_id)),
            "policies_to_train": ["ue_policy", "server_policy"],
        },
        "train_batch_size": 4000,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 10,
        "lr": 3e-4,
        "clip_param": 0.2,
        "lambda": 0.95,
        "model": {"vf_share_layers": True},
    }

    stop = {"training_iteration": args.stop_iters}
    tune.run("PPO", config=config, stop=stop, local_dir=args.logdir, checkpoint_at_end=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--stop-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slot-time", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--logdir", type=str, default="./rllib_results")
    args = parser.parse_args()
    main(args)
