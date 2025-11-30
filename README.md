# RealTimeSystemsProject
MEC MAPPO Phase-1 (RLlib) 

1) Create a virtualenv and install dependencies:
   python -m venv venv
   source venv/bin/activate
   pip install -r reqs.txt

2) Train (multi-agent PPO as MAPPO proxy):
   python train_mapppo.py --num-cpus 4 --N 5 --stop-iters 200 --logdir ./rllib_results

3) Evaluate a checkpoint:
   python eval_and_plot.py --checkpoint <PATH_TO_CHECKPOINT> --N 5 --episodes 10

Notes:
 - Checkpoint path is printed in the Tune output directory (./rllib_results/...).
 - Tweak hyperparameters in train_mapppo.py if you want more stable training (train_batch_size, lr, workers).
 - For debugging, use N=2 or 3 and stop-iters small (10) to ensure things run.

