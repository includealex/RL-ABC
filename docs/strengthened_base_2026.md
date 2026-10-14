# Strengthened RL-ABC base: design, provenance, and migration guide

This document explains the consolidation performed in
`RL-ABC-clean-code-2026`. It describes the previous behavior, the current
behavior, where each change came from, and what the change means for future
RL-ABC tasks.

## 1. Goal and scope

The repository is intended to be the common base for injection, extraction,
matching, QDAC, and other accelerator-control tasks. The consolidation follows
one rule:

> Shared parsing, simulation, observation, action, logging, and reliability
> behavior belongs in the base. Task-specific lattices, generated particles,
> rewards, archives, and algorithms remain in their task repositories.

The source repositories used during the comparison were:

| Source | What was reused |
|---|---|
| `RL-ABC-clean-code-2026` before this update | Portable configuration, pinned environment, runtime checks, Gymnasium API behavior, float32 action spaces, visualization helpers, and the clean repository layout. |
| `RLABC_QDAC` | The enhanced Elegant wrapper, generated-lattice validation, particle-derived beam-optics analysis, target comparison, descriptors, and optional combined matching reward. |
| `slow extraction` | The hardened lattice parser, occurrence-specific action registry, dynamic observation sizing, angle-and-tilt context, explicit generated/external beam modes, isolated evaluation outputs, and newer failure handling. |
| `slow extraction/toy_slow_extraction` | Used as an integration reference for external SDDS input and warning behavior. Its task-specific generator, inputs, outputs, and configuration were not copied. |

The original comparison folders were not modified.

## 2. High-level data flow

The environment now has one traceable path from lattice input to policy input:

```text
source .lte + source .ele
        |
        v
seed-local tracking file and working lattice
        |
        v
structural lattice expansion
  - comments remain comments
  - nested/repeated/reversed LINE items are resolved
  - repeated controllable occurrences receive unique names
  - a coordinate WATCH is inserted before every controllable occurrence
        |
        +------------------------------+
        |                              |
        v                              v
magnet action registry          Elegant validation
  - exact parameters              - internal structure checks
  - policy masks                  - real Elegant run_setup parse
  - full lattice indices
        |
        v
Elegant tracking -> WATCH SDDS file -> sdds2stream -> particle DataFrame
        |
        v
57 beam/aperture features + signed ANGLE + TILT
        |
        v
finite float32 observation with a checked width
```

At the default `n_bins=5`, the checked observation width is 59.

## 3. Observation: previous behavior

The previous clean environment intended to return 57 values:

```text
16 particle-distribution statistics
+ 25 x-y histogram bins
+ 1 particle-survival value
+ 1 aperture-containment value
+ 10 covariance values
+ 4 previous/next aperture values
= 57
```

There were three important inconsistencies:

1. The observation did not describe the upcoming magnet's bend angle or tilt.
2. The environment declared 57 values, but Elegant's fallback observation used
   an older 55-value layout. A simulator/SDDS failure could therefore return a
   vector with a different shape from `observation_space`.
3. The Elegant result reader passed `n_bins=5` directly, even when the
   environment had been configured with another value.

Those inconsistencies could cause a neural-network matrix-shape error far away
from the real source of the problem.

## 4. Observation: current behavior

### 4.1 Single size definition

`rl_framework.Utils.nn_observation_size()` is now the single source of truth:

```text
observation_width = 34 + n_bins**2
```

For `n_bins=5`:

```text
34 + 5**2 = 59
```

The environment space, normal observation builder, empty-beam observation,
Elegant fallback, reset validation, and step validation all use this function.

Changing `n_bins` is supported, but it changes the network input width. For
example, `n_bins=7` produces `34 + 49 = 83` values. The project requirement of
59 therefore means keeping `environment.n_bins` equal to 5.

### 4.2 Exact 59-value layout

For `n_bins=5`, zero-based indices are:

| Indices | Count | Meaning |
|---|---:|---|
| `0:16` | 16 | For each of `x`, `y`, `xp`, and `yp`: median, IQR, 10th percentile, and 90th percentile. |
| `16:41` | 25 | Flattened 5-by-5 `x-y` histogram, normalized by the number of surviving particles. |
| `41` | 1 | Surviving particle fraction: `surviving / initial`. |
| `42` | 1 | Fraction of particles inside the previous elliptical aperture. |
| `43:53` | 10 | Upper triangle of the standardized 4-by-4 covariance matrix for `x`, `y`, `xp`, and `yp`. |
| `53:57` | 4 | Previous `X_MAX`, previous `Y_MAX`, next `X_MAX`, and next `Y_MAX`. |
| `57` | 1 | Signed `ANGLE` of the upcoming dipole; zero for a quadrupole or terminal state. |
| `58` | 1 | `TILT` of the upcoming magnet; zero when absent or at the terminal state. |

The last two values are lattice context. They are appended even when the beam
DataFrame is empty, so loss of all particles does not erase which magnet was
being approached.

### 4.3 How ANGLE and TILT are found

Every generated WATCH is immediately before the controllable magnet that it
describes. `lattice_context_for_watch_point()` locates that WATCH in the
expanded graph and then finds the upcoming magnet.

The rules are:

- `QUAD` and `KQUAD`: `ANGLE=0`; use the lattice's `TILT` if present.
- Supported dipole families (`SBEND`, `CSBEND`, `RBEND`, `KSBEND`, and the
  other aliases listed in `Utils_magnet_actions.py`): use the lattice's
  `ANGLE` and `TILT`. If `TILT` is absent, it is explicitly represented as
  zero.
- A reversed occurrence such as `-B1` negates `ANGLE`.
- `TILT` is retained exactly as specified; it is not negated by line reversal.
- The final WATCH has no following magnet, so its context is `(0, 0)`.
- Invalid numeric `ANGLE` or `TILT` text raises a clear error instead of being
  silently converted to zero.

This is not a separate categorical magnet-type feature. Magnet family is
represented implicitly: quadrupoles have zero angle, while ordinary bending
dipoles have a nonzero signed angle. Adding an explicit type flag would produce
60 values and would require a new observation schema and new networks.

### 4.4 Verified physical example

The actual example lattice was run through Elegant. The observed contexts were:

```text
reset / quadrupole:  ANGLE=0,          TILT=0
WBM1_1 / dipole:     ANGLE=-0.785398,  TILT=0
WBM2_1 / dipole:     ANGLE=-0.785398,  TILT=0
```

This proves that the feature is not only appended in a unit-test helper; it
survives lattice preprocessing, Elegant execution, SDDS conversion, observation
construction, and environment output.

## 5. Lattice preprocessing

### 5.1 Previous behavior

The old preprocessing relied heavily on line-oriented regular expressions and
assumptions about `QUAD` and `SBEND`. This made several cases fragile:

- multiline definitions;
- comments containing text that looked like a definition;
- nested `LINE` definitions;
- `N*ELEMENT` repetition;
- reversed `-ELEMENT` references;
- the same controllable definition used multiple times;
- Elegant element aliases such as `KQUAD` and `CSBEND`.

Variable ordering also used substring replacement such as removing `K1` or
`FSE` from names. An element whose own name contained one of those strings
could be interpreted incorrectly.

### 5.2 Current behavior

The hardened parser adapted from slow extraction is now the canonical
`rl_framework/Utils.py` implementation. It:

1. separates active code from `!` comments;
2. parses logical statements instead of assuming one definition per line;
3. structurally expands the selected beamline;
4. supports repetition and reversal;
5. clones repeated controllable occurrences as unique names;
6. inserts one coordinate-mode WATCH immediately before each active
   controllable occurrence;
7. rewrites only control parameters that actually exist (`K1`, `HKICK`,
   `VKICK`, or `FSE`);
8. preserves other parameters, including `ANGLE` and `TILT`;
9. validates uniqueness, references, WATCH placement, and graph structure.

The generated lattice is also optionally passed to Elegant itself using a
minimal `run_setup` input. This catches syntax that an internal parser might
accept but Elegant rejects.

## 6. Action handling and masks

### 6.1 What stays fixed

The neural-network policy still has four outputs:

```text
[K1, HKICK, VKICK, FSE]
```

Keeping a fixed policy width is important for DDPG and Stable-Baselines3.

### 6.2 What changed

Previously, the environment inferred action layout from variable counts and
common cases such as three quadrupole controls or one dipole control. The new
`MagnetActionRegistry` is built from the fully expanded active lattice.

For every magnet occurrence it records:

- unique occurrence name;
- normalized Elegant element type;
- quadrupole or dipole family;
- control parameters explicitly present on that occurrence;
- names and indices in the full beamline variable vector;
- availability, reset-policy, and effective policy masks.

Examples:

```text
QUAD with K1, HKICK, VKICK -> availability [1, 1, 1, 0]
dipole with FSE            -> availability [0, 0, 0, 1]
```

When `reset_specific_keys_bool=true`, the channels listed in
`environment.reset_keys` are disabled. The backward-compatible default is:

```json
"reset_keys": ["HKICK", "VKICK", "FSE"]
```

With that default:

```text
QUAD effective mask  -> [1, 0, 0, 0]
FSE-only bend mask   -> [0, 0, 0, 0]
```

The list is case-insensitive, duplicate entries are removed, and its canonical
order is always `[K1, HKICK, VKICK, FSE]`. Any subset is valid:

```text
// Disable only K1
"reset_keys": ["K1"]

// Disable the two steering kicks
"reset_keys": ["HKICK", "VKICK"]

// Disable K1 and dipole field-scale error
"reset_keys": ["K1", "FSE"]

// Disable all four policy channels
"reset_keys": ["K1", "HKICK", "VKICK", "FSE"]

// Disable no channels even when the boolean is true
"reset_keys": []
```

The examples above are fragments; JSON configuration files cannot contain
comments. An unsupported value such as `ANGLE` fails during environment
construction rather than silently doing nothing.

If `reset_specific_keys_bool=false`, `reset_keys` is retained in the
configuration but no channel is disabled. This makes it safe to switch the
policy on and off without rewriting the desired list.

The registry maps the four policy outputs into the correct positions of the
full lattice vector and retains values already chosen for earlier occurrences.
It does not invent parameters that are absent from the lattice definition.

For the included full machine lattice, the verified inventory contains:

```text
15 controllable magnet occurrences
37 full lattice control variables
16 WATCH observations including final_WP
```

## 7. Particle-source handling

### 7.1 Previous behavior

The clean environment assumed a generated `&bunched_beam` and attempted to
edit `n_particles_per_bunch`. That was not reliable for slow extraction, where
the particle source may be an external SDDS file loaded through `&sdds_beam`.
The configured count could disagree with the actual file.

### 7.2 Current behavior

The `environment.set_initial_particles` setting selects an explicit mode.

Generated mode:

```json
{
  "set_initial_particles": true,
  "init_num_particles": 1000
}
```

- requires exactly one active `&bunched_beam`;
- requires a positive integer `init_num_particles`;
- changes `n_particles_per_bunch` only in the seed-local `.ele` copy.

External SDDS mode:

```json
{
  "set_initial_particles": false
}
```

- requires exactly one active `&sdds_beam`;
- reads its `input` path relative to the `.ele` file;
- counts the SDDS `x` column with `sdds2stream`;
- treats that count as authoritative and ignores `init_num_particles`.

A deck containing both sources, no source, or a mode/source mismatch fails
before training. The original `.ele` file remains unchanged; lattice path,
results path, particle count, and Elegant random seed updates occur in a
seed-specific copy.

## 8. Elegant wrapper

### 8.1 Core wrapper changes

The core wrapper in `rl_framework/Elegant.py` includes the general-purpose
slow-extraction fixes and retains the clean repository's plotting interface.
Important changes include:

- configurable `n_bins` propagated into observation construction;
- 59-value fallback through the shared size function;
- occurrence-aware registry construction;
- SDDS return-code checking with the failing filename in the error;
- argument-list subprocess execution instead of interpolating the normal
  Elegant/SDDS command through a shell;
- execution from the tracking file's directory, which makes absolute input
  paths work without corrupting Elegant's `%s` output root name;
- seed-local preprocessing, so parallel jobs do not mutate the shared source;
- safe Plotly `show=False` behavior retained from the clean base.

If an Elegant simulation fails during `step()`, the environment no longer asks
the wrapper to parse possibly stale result files. A reset retries the simulator
three times and then raises the actual failure instead of continuing into a
later divide-by-zero or shape error.

### 8.2 QDAC-derived enhanced wrapper

`rl_framework/Elegant_edited.py` subclasses the core wrapper. The environment
and classical optimizers now use this enhanced wrapper.

It adds:

- cached validation of the generated lattice by the configured Elegant binary;
- `sddsanalyzebeam` analysis of coordinate-mode WATCH files;
- optional momentum filtering through `sddsprocess`;
- uncorrected and dispersion-corrected beam quantities;
- optics-target aliases such as `beta_x -> betax`;
- normalized residual descriptors suitable for QDAC;
- a bounded optics score;
- an optional combined transmission/optics reward.

For target quantity `i`:

```text
normalized_error_i = (measured_i - target_i) / scale_i
loss = weighted mean(normalized_error_i**2)
score = 1 / sqrt(1 + loss)
```

The score lies in `(0, 1]`, with 1 representing an exact match.

This functionality is available but does not silently change the base
environment reward. The existing survival reward remains the default.
Task-specific QDAC code may explicitly call `compare_watch_to_target()` or
`evaluate_watch_match()`.

## 9. Environment reliability

The merged environment keeps the Gymnasium interface:

```python
observation, info = env.reset(seed=seed)
observation, reward, terminated, truncated, info = env.step(action)
```

Additional checks now guarantee that every returned observation:

- is flattened;
- uses `float32`;
- has the width calculated from `n_bins`;
- contains only finite values.

The real and compatibility action spaces also use `float32`. Reset calls
`super().reset(seed=seed)`, so Gymnasium's random-number behavior remains
correct.

The package `__init__.py` now lazily imports the environment and DDPG agent.
Tools that only need a utility function no longer initialize Gymnasium, Torch,
HoloViews, and the accelerator interface during package import.

## 10. DDPG and replay compatibility

### 10.1 Why old artifacts cannot be loaded directly

An actor trained on a 57-value state has a first layer shaped for 57 inputs.
The strengthened environment supplies 59. Loading that state dictionary without
a check would eventually produce a matrix multiplication or parameter-shape
error.

The DDPG loader now inspects `fc1.weight` before loading:

- 59 input columns: accepted for the default schema;
- 57 or 58 input columns: rejected with an observation-schema explanation.

Replay-buffer loading similarly checks the stored current and next state widths.
Legacy five-field replay transitions are also rejected because the current
agent requires current and next action masks.

Starting a new model and replay buffer is the recommended migration. Padding an
old network with two untrained columns is technically possible, but it is not
performed automatically because that is a modeling decision, not a safe file
conversion.

### 10.2 Logging and resume behavior

The slow-extraction run-isolation changes were retained:

```text
logs_<seed>/run_<timestamp>_pid<process>/...
plots_<seed>/run_<timestamp>_pid<process>/...
```

This prevents concurrent or repeated evaluations from overwriting one another.
Checkpoint state now includes observation width, episode count, evaluation
count, run identifier, step count, and best evaluation result. Loading remains
controlled by `train.py`; the agent constructor does not load the same
checkpoint a second time.

## 11. Configuration changes

The default environment section now includes:

```json
{
  "stage": null,
  "n_bins": 5,
  "reset_specific_keys_bool": false,
  "reset_keys": ["HKICK", "VKICK", "FSE"],
  "init_num_particles": 1000,
  "set_initial_particles": true,
  "validate_generated_lattice": true
}
```

The settings are exposed through command-line overrides:

```bash
python train.py --environment.set_initial_particles false
python train.py --environment.validate_generated_lattice false
python train.py --environment.reset_specific_keys_bool true \
  --environment.reset_keys K1 FSE
```

Lattice validation should normally remain enabled. Disabling it is intended for
parser/unit tests or an explicitly controlled workflow without an Elegant
binary.

## 12. Warning and visualization fixes

The merge also retained or added the following smaller reliability fixes:

- regex separators use raw strings, eliminating invalid-escape warnings;
- all project Python files compile with warnings promoted to errors;
- the notebook HTML iframe suppresses only IPython's known iframe suggestion;
- `plot_mag_3d(..., show=False)` again returns a Plotly figure without opening
  a display;
- evaluation logs are not mixed into the noisy training CSV;
- result and log directories are created recursively when needed.

## 13. What was intentionally not copied

The strengthened base does not import everything from the comparison folders.
The following remain task-specific:

- QDax/JAX emitters, repertoire/archive management, and QDAC training loops;
- task-specific optical targets and reward weights;
- slow-extraction particle generators and generated SDDS input files;
- experiment result folders, logs, plots, checkpoints, and replay buffers;
- task-specific lattice and training YAML files.

This keeps the base usable for DDPG, Stable-Baselines3, classical optimization,
and future QDAC tasks without forcing every installation to carry QDax/JAX or a
particular beam source.

## 14. Verification performed

The consolidated implementation was checked at several levels:

1. **Static compilation:** every Python source compiled with warnings treated
   as errors.
2. **Regression suite:** 24 tests passed, covering the 59-value layout,
   dynamic histogram size, quadrupole/dipole aliases, reversal, missing-tilt
   fallback, invalid lattice values, particle-source modes, configurable reset
   masks, wrapper-side zeroing, replay compatibility, enhanced wrapper
   inheritance, and visualization behavior.
3. **Lattice preprocessing smoke test:** 104 expanded occurrences, 16 WATCH
   points, 15 controllable occurrences, and 37 variables were validated.
4. **Real Elegant/SDDS setup check:** executable paths, input files, package
   versions, and `defns.rpn` passed.
5. **Real Elegant environment test:** reset and step both returned finite
   `(59,)` float32 observations. A second integration check configured
   `reset_keys=["K1"]`, verified the reset mask `[0, 1, 1, 1]`, held `Q1L0K1`
   at zero, and still applied the requested `HKICK` and `VKICK` values.
6. **Physical context test:** real dipole WATCH observations contained the
   expected signed angle.
7. **DDPG integration test:** a real one-step transition used a 59-value state,
   four-value policy action, two four-value masks, and was stored successfully
   in replay memory.

All full simulation tests ran in temporary directories. No generated Elegant
inputs, results, logs, plots, observations, models, or replay buffers were added
to the repository.

## 15. Guidance for future task variants

When creating a new task from this base:

1. Keep `n_bins=5` if the task must remain on the 59-value schema.
2. Start a new checkpoint and replay buffer unless the source artifacts already
   use the same observation and action-mask schemas.
3. Set `set_initial_particles` to match the active `.ele` source namelist.
4. Keep the source lattice immutable; let preprocessing create seed-local files.
5. Put `ANGLE` and `TILT` in the lattice, not in task-specific observation code.
6. Add controls only by defining the supported parameter on the actual magnet;
   the registry will discover it.
7. Keep general fixes in `rl_framework`; keep rewards, targets, generators, and
   experiment configurations in the task directory.
8. Run the unit suite and a one-step real Elegant smoke test before long
   training.

The most important invariant is:

> The lattice graph, action registry, Elegant wrapper, environment observation
> space, neural-network input, replay buffer, and checkpoint must all describe
> the same schema.
