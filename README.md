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

### Reproducing results locally

There are two ways to reproduce the simulations:

#### Option 1 — Using the Bash script (`run_simus.sh`)

Use this option if you want to repeat the experiment several times automatically (for different random seeds or parameter configurations).

```bash
bash run_simus.sh
```

The bash script loops over multiple seeds or settings defined inside the file, allowing batch execution of experiments — for example, to compute Monte Carlo averages as in the paper.
You can edit the bash file to change:
- the model (`--model m1`, `--model m2`, `--model m3` for models 1, 2, and 3 from the paper)
- the number of trajectories (`--N`),
- or the range of seeds.

#### Option 2 — Using the Python script directly

Use this option to run a single experiment manually:

```bash
python main_simus.py --model [MODEL_NAME] --N [NUM_TRAJECTORIES] --seed [SEED]
```

Example:

```bash
python main_simus.py --model m1 --N 200 --seed 42
```

This produces a single result file such as:

```bash
res/resN200_42_m1.pickle
```

>[!NOTE]
> Each `.pickle` file corresponds to one experiment.
> When using the bash script, several such files will be created automatically.


## Reproducing Paper Results

### Synthetic data experiments

To reproduce the tables and figures from the paper:

1. Run all simulation scripts to generate `.pickle` results.
2. Summarize results by model using:
```bash
python compile_results.py --model [MODEL_NAME]
```
This script collects all `.pickle files` corresponding to the specified model (e.g. `m1`) and aggregates them into a single summarized results file stored in the same directory.
3. Generate plots with:
```bash
python plot_results.py
```
> [!NOTE]
> `compile_results.py` must always be executed before `plot_results.py`.

### Real data application

The script `examples/Neuronal-application.ipynb` contains the code to reproduce the analysis of neuronal data presented in the paper. It includes data loading, preprocessing, model fitting, and visualization of results.