# RL-ABC: Reinforcement Learning for Accelerator Beamline Control

RL-ABC (RLABC) converts Elegant beamline lattice and command files into a
Gymnasium environment for reinforcement-learning-based accelerator tuning.
This repository contains the runnable framework used for:

> A. Ibrahim, F. Ratnikov, M. Kaledin, A. Petrenko, and D. Derkach,
> “RL-ABC: Reinforcement Learning for Accelerator Beamline Control,”
> *Computer Physics Communications* (2026).

The repository intentionally contains source code and required example inputs
only. Training logs, model checkpoints, replay buffers, generated Elegant
files, observations, plots, and optimization results are excluded and covered
by `.gitignore`.

## Repository contents

```text
.
├── train.py
├── sb_train.py
├── config_manager.py
├── config.json
├── machine.lte
├── track.ele
├── requirements.txt
├── rl_framework/
│   ├── __init__.py
│   ├── Environment.py
│   ├── Elegant.py
│   ├── Utils.py
│   └── Agents/
│       ├── __init__.py
│       └── DDPG.py
└── classical_optimizers/
    ├── README.md
    ├── scipy_optimization.py
    └── bayesian_optimization.py
```

## Prerequisites

- Python 3.10
- Elegant and its SDDS Toolkit

RLABC calls the external `elegant`, `sdds2stream`, `sddsquery`, and
`sddsprocess` executables. Install Elegant from the
[Advanced Photon Source software page](https://www.aps.anl.gov/Accelerator-Operations-Physics/Software).

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Edit `config.json` before running:

```json
{
  "platform": {
    "os_type": "linux",
    "elegant_path": "/path/to/elegant/bin/",
    "sdds_path": "/path/to/elegant/defns.rpn"
  }
}
```

Keep the trailing slash on `elegant_path`. Use `os_type: "darwin"` on macOS.
For an HPC command such as `srun -n 2 Pelegant`, set
`simulation.override_dynamic_command` to `true` and configure
`simulation.overridden_command`.

## Training

The default configuration starts a fresh, all-variable DDPG run with 1,000
particles, 10,000 episodes, and Gaussian exploration noise decaying from 0.5
to 0.05:

```bash
python train.py
```

Common overrides:

```bash
python train.py --training.seed 1 --training.cpu true
python train.py --training.n_episodes 100 --environment.init_num_particles 1000
```

The paper’s 30,000-episode from-scratch schedule can be reproduced in two
phases so that the noise reaches 0.05 after the first 10,000 episodes and then
stays there:

```bash
python train.py
python train.py \
  --training.load_model true \
  --buffer.load_buffer_bool true \
  --training.n_episodes 20000 \
  --training.greedy 0.05
```

The first command creates the checkpoint and replay buffer used by the second
command. These generated files remain local and are not tracked by Git.

For K-only training, set `environment.reset_specific_keys_bool` to `true`.
Setting it to `false` activates all 37 control variables: quadrupole strengths,
horizontal and vertical kicks, and dipole field-scaling errors.

## Stable-Baselines3

```bash
python sb_train.py --algo SAC --total-timesteps 100000 --seed 0
python sb_train.py --algo TD3 --total-timesteps 100000 --seed 0
python sb_train.py --algo DDPG --total-timesteps 100000 --seed 0
```

## Classical baselines

Differential evolution:

```bash
python classical_optimizers/scipy_optimization.py --seed 0
```

Bayesian optimization:

```bash
python classical_optimizers/bayesian_optimization.py \
  --backend bayes_opt \
  --n-calls 500 \
  --seed 0
```

The Bayesian script also supports `--backend skopt`.

## Required inputs and generated files

`machine.lte` is the example VEPP-5-derived beamline with 11 quadrupoles and
four dipoles. `track.ele` is the matching Elegant command template. Runtime
code makes seed-specific copies of these inputs and writes simulation outputs
to generated directories such as `results_0/`, `inputs_0/`, and `observs_0/`.

Do not commit runtime artifacts. The ignore rules cover:

- `logs*`, `models*`, `results*`, `observs*`, `inputs*`, `plots*`, and
  TensorBoard summaries;
- checkpoints, replay buffers, and optimizer result files;
- generated Elegant lattice, tracking, SDDS, and CSV files.

## License

MIT License. See `LICENSE`.
