# ShapeOperatorLearning

Neural operator for predicting wall shear stress (WSS) and pressure on vascular geometries using shape coefficients as branch-net inputs.

## Overview

The model is a DeepONet-style architecture with a **branch net** (encodes shape via SVD/spherical-harmonic coefficients) and a **trunk net** (encodes query point coordinates). Training data comes from CFD simulations on parametric or TAA (thoracic aortic aneurysm) geometries.

## Repository structure

```
inputs/          # YAML configuration files
dataset/         # Dataset classes
models/          # Neural network definitions and training loops
utils/           # Geometry helpers, zero-case filters
train.py         # Main training entry point
predictionVisualize_param.py   # Visualization for parametric (spheroid) cases
predictionVisualize_TAA.py     # Visualization for TAA cylinder cases
```

## Training

```bash
python train.py \
  --config inputs/config_TAA.yaml \
  --coefficients ../TAA_CFD_pipeline/coefficient_data.npz
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `./inputs/config.yaml` | Path to YAML config |
| `--coefficients` | *(required)* | Path to `coefficient_data.npz` from the data pipeline |
| `--skip_zero_filter` | off | Skip the hardcoded zero-stress case filter (use for non-Nov6 datasets) |

### Provided configs

| Config | Use case |
|---|---|
| `inputs/config.yaml` | Parametric (Nov6) dataset, WSS |
| `inputs/config_TAA.yaml` | TAA dataset, WSS |
| `inputs/config_TAA_pressure.yaml` | TAA dataset, pressure (`out_dim: 1`) |

Outputs (model checkpoint, training curves, copied config) are written to `result_dir` defined in the config.

## Visualization

### TAA cases (cylindrical coordinates)

```bash
python predictionVisualize_TAA.py \
  --config inputs/config_TAA.yaml \
  --coefficients ../TAA_CFD_pipeline/coefficient_data.npz \
  --data-index 350 \
  --result-dir ./results_TAA/coeff_b24_64_t3_64_n600/
```

Produces three figures in `<result_dir>/plots/`:
- `unrolled_wss_case_<N>.png` — WSS components + magnitude on the unrolled (θ, z) cylinder
- `streamlines_case_<N>.png` — WSS streamlines on the unrolled surface
- `3d_wss_case_<N>.png` — 3D cylinder surface coloured by WSS magnitude

### Parametric (spheroid) cases

```bash
python predictionVisualize_param.py \
  --config inputs/config.yaml \
  --coefficients coefficient_data.npz \
  --data-index 2300 \
  --result-dir ./param_results_coeff/coeff_b3_128_t2_256_f128/
```

Produces per-case figures in `<result_dir>/plots/`:
- `shear_stress_components_case_<N>.png` — 3D sphere coloured by each WSS component
- `streamlines_case_<N>.png` — streamlines in (θ, φ) space

### Common visualization arguments

| Argument | Description |
|---|---|
| `--config` | Path to the same YAML used for training |
| `--coefficients` | Path to `coefficient_data.npz` |
| `--data-index` | Case index to visualize (must be in the validation range) |
| `--result-dir` | Directory with the saved `shear_stress_model.pt`; plots saved here too |

## Configuration reference

Key fields in any `inputs/config_*.yaml`:

```yaml
data:
  npz_files_pattern: path/to/data_{}.npz   # {} is replaced by the case index
  npz_range: [0, 599]                       # inclusive range of case indices
  ref_xyz_vtk_path: path/to/reference.vtk   # LDDMM reference mesh
  value_key: transformed_values             # NPZ key for target field (pressure_values for pressure)
  train_split: 0.8

model:
  branch_dims: [24, 64, 64, 64]   # first dim overridden by actual coefficient size
  trunk_dims:  [3,  64, 64, 64]
  final_dim: 64
  out_dim: 3                       # 3 for WSS vector, 1 for scalar pressure

training:
  n_epochs: 50
  learning_rate: 0.001
  batch_size: auto                 # one shape per batch

output:
  result_dir: ./results_TAA/run_name/
```
