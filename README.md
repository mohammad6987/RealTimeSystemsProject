# RealTimeSystemsProject

Code is organized by phase:

- `phase1/`: Phase-1 environment, models, and trainer.
- `phase2/`: Phase-2 hierarchical environment, models, and trainer.
- `main_hub.py`: single entry point to run either phase.

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r reqs.txt
```

## Run with Main Hub

Phase 1:

```bash
python main_hub.py phase1 --n-values 2 3 4 5 6 7 --iterations 120 --max-steps 200
```

Phase 2:

```bash
python main_hub.py phase2 --cluster-values 4 5 6 7 8 9 10 --n-users 100 --iterations 120 --max-steps 200
```

## Direct Trainers

```bash
python phase1/train_phase1_mappo.py
python phase2/train_phase2_hierarchical_mappo.py
```

## Output Paths

- Phase 1: `plots/phase1/N_<N>/...`
- Phase 2: `plots/phase2/K_<K>/...`
