# RL-ABC: Reinforcement Learning for Accelerator Beamline Control

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anwar9Ibrahim/RL-ABC/blob/clean-code-2026/notebooks/RLABC_GitHub_Colab_quickstart.ipynb)

RL-ABC (RLABC) converts Elegant lattice and command files into a Gymnasium
environment for reinforcement-learning-based accelerator tuning.

> A. Ibrahim, F. Ratnikov, M. Kaledin, A. Petrenko, and D. Derkach,<br>
> “RL-ABC: Reinforcement Learning for Accelerator Beamline Control,”<br>
> *Computer Physics Communications* (2026).

This branch contains source code and the required example inputs only. Generated
logs, checkpoints, replay buffers, observations, Elegant outputs, and plots are
excluded by `.gitignore`.

## Tested environment

RL-ABC is tested with **Python 3.10.19**. Use Python 3.10; Python 3.14 is not
supported by this release. The direct dependencies used in testing are pinned
with exact `==` versions in `requirements.txt`, including:

- PyTorch 2.10.0
- Gymnasium 1.2.3
- Stable-Baselines3 2.7.1
- NumPy 2.2.6
- pandas 2.3.3
- SciPy 1.15.2
- scikit-learn 1.7.2
- Bayesian Optimization 3.3.0
- scikit-optimize 0.10.2

## Repository contents

```text
.
├── train.py
├── sb_train.py
├── config_manager.py
├── runtime_checks.py
├── config.json
├── machine.lte
├── machine_2dipole.lte
├── track.ele
├── environment.yml
├── requirements.txt
├── notebooks/
│   └── RLABC_GitHub_Colab_quickstart.ipynb
├── rl_framework/
│   ├── Environment.py
│   ├── Elegant.py
│   ├── Elegant_edited.py
│   ├── visulize.py
│   ├── visualization.py
│   ├── Utils.py
│   ├── Utils_magnet_actions.py
│   └── Agents/DDPG.py
└── classical_optimizers/
    ├── scipy_optimization.py
    └── bayesian_optimization.py
```

## Google Colab quickstart

The repository includes a self-contained Colab demonstration using a short
four-quadrupole FODO beamline. Use the **Open in Colab** button at the top of
this page to launch it directly from the `clean-code-2026` branch.

The notebook installs Elegant and SDDS, creates an isolated Python 3.10
environment, checks the RL-ABC setup, runs a short DDPG demonstration, and
generates Elegant-derived interactive visualizations. Its default budget is a
software check rather than a publication-quality benchmark.

## 1. Download and install

Clone the clean branch and enter the repository directory:

```bash
git clone --branch clean-code-2026 \
  https://github.com/Anwar9Ibrahim/RL-ABC.git
cd RL-ABC
```

The project environment is named `acc-elegant-rl`. Reproduce it with Conda:

```bash
conda env create -f environment.yml
conda activate acc-elegant-rl
python -m pip check
```

If the environment already exists, activate and update it:

```bash
conda activate acc-elegant-rl
python -m pip install -r requirements.txt
python -m pip check
```

Alternatively, create an isolated Python 3.10 virtual environment:

```bash
python3.10 --version
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

On Windows, activate with:

```powershell
.venv\Scripts\activate
```

`pip check` should print `No broken requirements found`.

### Reusable visualization commands

After a configured Elegant run, generate an interactive floor layout directly
from Elegant's `xyz.sdds` and `beamline.mag` output:

```bash
python -m rl_framework.visualization layout \
  --config config.json \
  --output beamline_layout.html
```

Generate the layout and the Twiss beam-envelope/aperture view from one Elegant
run:

```bash
python -m rl_framework.visualization overview \
  --config config.json \
  --output-dir beamline_overview
```

After training, evaluate the saved DDPG policy and write betatron, dispersion,
Twiss, and layout figures:

```bash
python -m rl_framework.visualization policy \
  --config config.json \
  --output-dir policy_visualizations \
  --load-model-tag last
```

The same operations are available as Python functions:

```python
from rl_framework.visualization import (
    plot_configured_beamline,
    plot_configured_optics,
    plot_trained_policy,
)
```

Existing notebooks can also use the historical notebook-facing class:

```python
from rl_framework.visulize import ElegantVisualizer

visualizer = ElegantVisualizer(env, magnet_settings)
mag_3d = visualizer.env.wrapper.get_mag_3d()
visualizer.env.wrapper.plot_mag_3d(mag_3d)

twiss, htmltwiss = visualizer.plot_twiss()
display(twiss)
```

## 2. Install Elegant and SDDS

Elegant is an external accelerator simulation program, not a Python package.
Install Elegant and the SDDS Toolkit from the
[Advanced Photon Source software page](https://www.aps.anl.gov/Accelerator-Operations-Physics/Software).

RL-ABC requires these files:

- `elegant`
- `sdds2stream`
- `sddsquery`
- `sddsprocess`
- `defns.rpn`

Edit the `platform` section of `config.json`:

```json
{
  "platform": {
    "os_type": "linux",
    "elegant_path": "/absolute/path/to/elegant/bin/",
    "sdds_path": "/absolute/path/to/elegant/defns.rpn"
  }
}
```

Keep the trailing slash on `elegant_path`. Use `"darwin"` for macOS. For an
HPC command such as `srun -n 2 Pelegant`, set
`simulation.override_dynamic_command` to `true` and configure
`simulation.overridden_command`.

Paths may also be supplied without editing the file:

```bash
python train.py --check-setup \
  --platform.os_type linux \
  --platform.elegant_path /absolute/path/to/elegant/bin/ \
  --platform.sdds_path /absolute/path/to/elegant/defns.rpn
```

## 3. Input files

The input files must be in the **same directory as `train.py`**. The repository
includes both beamline lattices used in the paper:

- `machine.lte` — four-dipole test-beamline lattice;
- `machine_2dipole.lte` — exact two-dipole experiment lattice; relative to the
  test lattice, BM2 and BM3 are equal-length drifts and the bend polarities of
  BM1 and BM4 are reversed;
- `track.ele` — Elegant tracking command file shared by both examples.

Run commands from the repository root (`cd RL-ABC`). To select the two-dipole
variant, set `simulation.input_beamline_file` to `machine_2dipole.lte`. To use
another beamline, place its `.lte` and `.ele` files beside `train.py` and update
these `config.json` values:

```json
{
  "simulation": {
    "input_beamline_file": "machine.lte",
    "input_beam_file": "track",
    "beamline_name": "machine"
  }
}
```

`input_beam_file` is the filename stem, so `track` refers to `track.ele`.

## Observation and action contract

With the default `environment.n_bins: 5`, every observation has **59 float32
values** in this fixed order:

```text
16 robust x/y/xp/yp statistics
+ 25 normalized x-y histogram bins
+ 1 surviving-particle fraction
+ 1 aperture fraction
+ 10 covariance values
+ 4 previous/next aperture values
+ 1 signed upcoming dipole ANGLE (0 for a quadrupole)
+ 1 upcoming magnet TILT
= 59
```

Reversed dipole occurrences negate the angle. Tilt keeps the value written in
the active lattice. The terminal observation uses `angle=0` and `tilt=0`.
Changing `n_bins` changes the width according to `34 + n_bins**2`; the runtime,
fallback state, observation space, checkpoints, and replay-buffer checks all use
that same function. Old 57- or 58-value checkpoints and buffers are rejected
with an explicit migration error.

The policy action stays fixed at `[K1, HKICK, VKICK, FSE]`. An
occurrence-specific registry masks channels that are absent from the active
magnet definition, including repeated magnets and Elegant quadrupole/bend
aliases.

## Strengthened Elegant wrapper

The environment uses the enhanced wrapper in `Elegant_edited.py`. It combines
the general lattice/action fixes with the QDAC wrapper's generated-lattice
validation and particle-derived optics comparison methods. Validation is on by
default (`environment.validate_generated_lattice: true`) and can be disabled
for parser/unit tests that intentionally do not have Elegant installed.

Beam-source handling is explicit:

- `set_initial_particles: true` requires exactly one `&bunched_beam` and writes
  `init_num_particles` into it.
- `set_initial_particles: false` requires exactly one `&sdds_beam`; the SDDS
  file is counted and is the authoritative initial particle count.
- Mixed or mismatched source modes fail before training.

For the full before/after explanation, source provenance, data flow, migration
notes, and verification evidence, see
[`docs/strengthened_base_2026.md`](docs/strengthened_base_2026.md).

## 4. Check the installation

Before starting a long run:

```bash
python train.py --check-setup
```

This checks:

- Python is version 3.10;
- the input `.lte` and `.ele` files exist;
- Elegant and the required SDDS executables exist;
- `defns.rpn` exists;
- the principal installed package versions.

The command exits without training and prints `RL-ABC setup check passed` when
the environment is ready.

## 5. Run a short end-to-end smoke test

This command trains from scratch for one short episode with 100 particles:

```bash
python train.py \
  --training.n_episodes 1 \
  --training.max_steps 1 \
  --environment.init_num_particles 100 \
  --training.cpu true \
  --agent.batch_size 2 \
  --agent.max_size 100
```

It exercises Elegant, the Gymnasium environment, the DDPG agent, evaluation,
replay-buffer saving, and checkpoint saving.

## 6. Train from scratch

The committed `config.json` is deliberately configured for a fresh run:

```json
{
  "training": {
    "load_model": false,
    "n_episodes": 10000,
    "greedy": 0.5
  },
  "buffer": {
    "load_buffer_bool": false
  }
}
```

After configuring the Elegant paths, start training with:

```bash
python train.py
```

No checkpoint or replay buffer is required. The old failure involving
`models_0/ddpg_checkpoint_last.pth` cannot occur in the default from-scratch
configuration because checkpoint loading is disabled.

For another seed:

```bash
python train.py --training.seed 1
```

For K1-only training, enable reset behavior and keep the default reset list:

```bash
python train.py --environment.reset_specific_keys_bool true
```

The default is:

```json
"reset_keys": ["HKICK", "VKICK", "FSE"]
```

Any subset of `K1`, `HKICK`, `VKICK`, and `FSE` can be disabled. For example,
to hold only K1 and FSE at zero:

```bash
python train.py \
  --environment.reset_specific_keys_bool true \
  --environment.reset_keys K1 FSE
```

To disable every policy channel:

```json
{
  "reset_specific_keys_bool": true,
  "reset_keys": ["K1", "HKICK", "VKICK", "FSE"]
}
```

For all 37 control variables, keep
`environment.reset_specific_keys_bool` set to `false`.

## Paper training schedule

The 30,000-episode from-scratch schedule consists of 10,000 episodes while the
Gaussian exploration scale decays from 0.5 to 0.05, followed by 20,000
episodes at 0.05:

```bash
# Phase 1: starts from scratch and creates the checkpoint and replay buffer
python train.py

# Phase 2: explicitly continues from the files created by phase 1
python train.py \
  --training.load_model true \
  --buffer.load_buffer_bool true \
  --training.n_episodes 20000 \
  --training.greedy 0.05
```

Only set `training.load_model` or `buffer.load_buffer_bool` to `true` when the
corresponding local file exists. If a requested file is absent, RL-ABC now
reports its exact path and explains how to start from scratch.

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

## Generated files

Training creates seed-specific directories such as `models_0/`, `logs_0/`,
`plots_0/`, `results_0/`, `inputs_0/`, and `observs_0/`. It also creates
checkpoints, replay buffers, CSV logs, and temporary Elegant/SDDS files. These
files remain local and are ignored by Git.

## Troubleshooting

### Unsupported Python

If RL-ABC reports an unsupported Python version, recreate the environment with
Python 3.10. Do not reuse a Python 3.14 virtual environment.

### Setup check fails

Run `python train.py --check-setup`. The message lists each missing input,
executable, or SDDS definitions file.

### Checkpoint or replay buffer is missing

Train from scratch:

```bash
python train.py \
  --training.load_model false \
  --buffer.load_buffer_bool false
```

### Elegant simulation fails

Verify the configured paths, then run the exact generated `.ele` command
directly to inspect Elegant’s error output. Also confirm that `machine.lte` and
`track.ele` are beside `train.py`.

## License

MIT License. See `LICENSE`.
