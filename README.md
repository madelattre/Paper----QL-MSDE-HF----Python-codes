# Numerical experiments on "quasi-likelihood approach for mixed-effects stochastic differential equation models with high-frequency data"

This repository contains the Python code to reproduce the numerical experiments from the paper:

> **Quasi-likelihood inference for SDE with mixed-effects observed at high frequency.**
> Authors: Delattre, M., & Masuda, H. (2025).  
> arXiv preprint arXiv:2508.17910.

This repository provides the simulation and estimation framework used in our study on quasi-likelihood inference for mixed-effects stochastic differential equations (SDEs) observed at high frequency.
The goal is to evaluate the finite-sample performance of the proposed estimators under various model settings.

---

## Overview

- Python module `mixedsde` for defining mixed SDE models as instances of a model class `Mixedsde`.  
- Scripts for simulating trajectories and estimating parameters using maximum likelihood methods.  
- Bash script (`run_simus.sh`) for launching experiments locally or on a computing cluster.  
- Results are saved as `.pickle` files in the `res/` directory.

---

## Repository Structure

| File/Folder               | Description |
|----------------------------|------------|
| `mixedsde/`               | Python module defining the `Mixedsde` class and associated functions for drift, diffusion, and random effects. |
| `main_simus.py`           | Main Python script to run simulations on parameter estimation. |
| `run_simus.sh`            | Bash script to launch experiments. Modify `--model` and `--N` arguments for different experiments. |
| `res/`                    | Directory where `.pickle` result files are saved. Can be changed in `main_simus.py`. |
| `examples/` | Scripts and materials for reproducing the real-data analysis (neuronal data application) presented in the paper. |
---

## Installation

To recreate the environment:

```bash
conda env create -f environment.yml
conda activate mixedsde_py312
```

## Usage

### Running simulations locally

Modify the bash script to choose your model and number of trajectories:

```bash
bash run_simus.sh
```

Or run directly with Python:

```bash
python main_simus.py --model [MODEL_NAME] --N [NUM_TRAJECTORIES] --seed [SEED]
```

In the bash file or command line, the important arguments are
- `--model`: Name of the model to simulate: `m1`, `m2`, `m3` for models 1, 2, and 3 from the paper.
- `--N`: Number of trajectories to simulate.
- Results are saved as `.pickle` files in the `res/` directory. 

### Output

Each `.pickle` file corresponds to one experiment.

### Example

```bash
python main_simus.py --model m1 --N 200 --seed 42
```

produces a file such as: 

```bash
res/resN200_42_m1.pickle
```

## Reproducing Paper Results

### Synthetic data experiments

To reproduce the tables and figures from the paper:

1. Run all simulation scripts to generate `.pickle` results.
2. Assemble results using:
```bash
python compile_table_results.py
```
3. Generate plots with:
```bash
python plot_results.py
```
> [!NOTE]
> `compile_table_results.py` must always be executed before `plot_results.py`.

### Real data application

The script `examples/Neuronal-application.ipynb` contains the code to reproduce the analysis of neuronal data presented in the paper. It includes data loading, preprocessing, model fitting, and visualization of results.