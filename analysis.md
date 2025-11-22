````markdown
# Analysis of NNConv + PerforatedAI (PAI) on QM Dataset

## Table of Contents

- [1. Experiment Setup](#1-experiment-setup)
  - [1.1 Dataset and Task](#11-dataset-and-task)
  - [1.2 Model Architecture](#12-model-architecture)
  - [1.3 Training Configuration](#13-training-configuration)
- [2. Baseline vs PAI Summary](#2-baseline-vs-pai-summary)
- [3. Training Dynamics (seed = 60)](#3-training-dynamics-seed--60)
  - [3.1 Validation Curves and PAI Switch Points](#31-validation-curves-and-pai-switch-points)
  - [3.2 Learning Rate Schedule](#32-learning-rate-schedule)
  - [3.3 Epoch Time and Runtime Overhead](#33-epoch-time-and-runtime-overhead)
- [4. Cycle-Level Structural Analysis](#4-cyclelevel-structural-analysis)
  - [4.1 Parameter Counts per Cycle](#41-parameter-counts-per-cycle)
  - [4.2 Cycle-Level Performance](#42-cyclelevel-performance)
- [5. Discussion and Limitations](#5-discussion-and-limitations)
- [6. How to Reproduce](#6-how-to-reproduce)

---

## 1. Experiment Setup

### 1.1 Dataset and Task

The experiment evaluates a graph neural network (GNN) based on `NNConv` and `Set2Set` on a QM-style molecular property prediction task.

- Source file: `train.pt`
- Number of graphs: `N = 20,000`
- Node feature dimension: `num_features = 11`
- Edge feature dimension: `edge_in = 4`
- Task: **regression** – predict a scalar target `y` per graph.
- Evaluation metrics:
  - Training: Mean Squared Error (MSE)
  - Validation/Test: Mean Absolute Error (MAE), plus MSE for logging

Dataset split (fixed once using `CFG.seed = 60`):

- Train: 16,000 graphs
- Validation: 2,000 graphs
- Held-out test: 2,000 graphs

### 1.2 Model Architecture

Both **Baseline** and **PAI** use exactly the same backbone:

- Input projection: `Linear(num_features → dim)`
- **3 × NNConvBlock**, each block:
  - `NNConv(dim → dim, edge_mlp(edge_in → dim*dim), aggr='mean')`
  - `GraphNorm(dim)`
  - `Dropout(p=0.1)`
  - Residual connection around the block
- Global pooling: `Set2Set(dim, processing_steps=3)`
- Prediction head:
  - `Linear(2*dim → dim) → ReLU → Dropout → Linear(dim → 1)`

Key hyperparameters:

```text
dim           = 64
n_layers      = 3
dropout       = 0.1
batch_size    = 128
max_epochs    = 200
lr            = 5e-4
weight_decay  = 0.0
grad_clip     = 2.0
scheduler     = ReduceLROnPlateau(mode='min', factor=0.5,
                                  patience=5, min_lr=1e-6)
early_stop    = 20 epochs without validation improvement
````

### 1.3 Training Configuration

Reproducibility:

* Global seeds set for `python`, `numpy`, and `torch` (CPU + CUDA).
* Dataset split uses `torch.Generator().manual_seed(CFG.seed)`.
* Each `DataLoader` uses a seeded generator and a `worker_init_fn` to make workers deterministic.
* CUDA deterministic algorithms are enabled where supported.

PAI-specific configuration:

* `initialize_pai(doing_pai=True, save_name='PAI', making_graphs=False)`
* Switch mode: `DOING_HISTORY`
* Scheduled parameters (where available in the open-source build):

  * `n_epochs_to_switch = 10`
  * `p_epochs_to_switch = 10`
  * `history_lookback = 1`
  * Intended improvement threshold ≈ 0.003 for triggering structure changes.

All PAI artifacts for this run are saved under:

```text
outputs/PAI_seed60/
  ├── Scores.csv
  ├── Times.csv
  ├── LearningRate.csv
  ├── paramCounts.csv
  ├── switchEpochs.csv
  ├── best_test_scores.csv
  └── pai_plots.png
```

---

## 2. Baseline vs PAI Summary

The table below summarizes the main metrics for the **Baseline** and **PAI** models on the same train/val/test split (seed = 60).

| Model    | Best Val MAE | Test MAE (held-out) |
| -------- | ------------ | ------------------- |
| Baseline | **0.585139** | **0.644203**        |
| PAI      | **0.574468** | **0.632334**        |

Observations:

* PAI improves **validation MAE** by about `0.0107` (≈ **1.8 % relative**).
* PAI improves **test MAE** by about `0.0119` (also ≈ **1.8 % relative**).
* On this seed, PAI provides a consistent, small but clear gain in both validation and held-out test performance over the vanilla NNConv baseline.

---

## 3. Training Dynamics (seed = 60)

All plots in this section are generated from the CSVs in `outputs/PAI_seed60/`.
If you save them from the notebook, suggested filenames are:

* `plots/validation_curves_seed60.png`
* `plots/lr_schedule_seed60.png`
* `plots/epoch_time_seed60.png`

### 3.1 Validation Curves and PAI Switch Points

![Validation curves and PAI switch points](plots/validation_curves_seed60.png)

Data sources:

* `Scores.csv` (`epoch`, `val`, `running_val`, `test_at_valbest`)
* `switchEpochs.csv` (`epoch_switch_back_to_neuron`)

Key patterns:

* The **PAI validation MAE** drops quickly from ≈ 0.76 at the first epoch to ≈ 0.64 by ~epoch 20, then continues to improve more slowly toward ≈ 0.58.
* The **running-best curve** is smooth and monotonically decreasing; by the end of training it stabilizes around **0.575–0.576**.
* **Vertical dashed lines** mark PAI “switch back to neuron” epochs. In this run there are two visible switches, around **epoch 73** and **epoch 88**:

  * The first switch roughly coincides with a long plateau of the validation curve.
  * After the first switch, validation MAE briefly worsens, then recovers and slightly improves again.
* The global best validation MAE reported by the training loop is **0.574468**, occurring near the end of training after the last switch.

Interpretation:

* PAI is conservative: it allows the base architecture to learn for tens of epochs before attempting structural changes.
* Switches are aligned with **slow-improvement regions** of the validation curve, which is exactly where architecture search is most useful.
* After switching, the model does not collapse; it stays in roughly the same MAE range and eventually finds a slightly better optimum.

### 3.2 Learning Rate Schedule

![Learning rate schedule](plots/lr_schedule_seed60.png)

Data source:

* `LearningRate.csv` (`epoch`, `learning_rate`)

Behavior:

* Epochs **1–42**: learning rate fixed at **5e-4**.
* Around **epoch 43**, `ReduceLROnPlateau` triggers the first decay to **2.5e-4**, matching the early plateau in validation performance.
* A second decay further reduces LR to **1.25e-4`** in the early 70s, again following a period of limited improvement.
* After the **first PAI switch** (≈ epoch 73), the optimizer and scheduler are re-created:

  * LR is **reset to 5e-4**, giving the restructured network a chance to explore with a larger step size.
  * Later in this new cycle, LR decays again to **2.5e-4** once the post-switch validation curve plateaus.

Takeaways:

* Learning-rate decays line up well with plateaus in the validation MAE.
* When PAI restructures the model, the LR reset ensures the new architecture is not stuck in a tiny-step regime inherited from the old one.
* The combination of **LR schedule + PAI switches** effectively creates multiple “training phases” within a single run.

### 3.3 Epoch Time and Runtime Overhead

![Epoch time](plots/epoch_time_seed60.png)

Data source:

* `Times.csv` (`epoch`, `time_sec`)

Summary statistics (from the notebook):

* **Average epoch time (overall):** `7.776 s`
* **Average before first switch (epoch < 73):** `7.713 s`
* **Average after first switch (epoch ≥ 73):** `8.079 s`

Observations:

* Per-epoch times fluctuate between roughly **6.8 s** and **9.0 s**, likely due to noise from data loading and GPU scheduling.
* After the first PAI switch, the average epoch time increases by about **0.37 s** (~**4–5 % overhead**).
* There is **no catastrophic slowdown**; the cost of using PAI is modest compared to overall training time.

Interpretation:

* For this setup, PAI’s structural updates add a small but acceptable runtime overhead.
* Given a ≈1.8 % improvement in MAE and only ≈5 % increase in per-epoch time after switching, the trade-off is reasonable.

---

## 4. Cycle-Level Structural Analysis

PAI maintains the concept of **cycles**: each time it switches back to neuron mode with a potentially restructured architecture, a new cycle starts.

The following analysis uses:

* `paramCounts.csv` – parameter count at the start of each cycle.
* `best_test_scores.csv` –, for each cycle, `(param_count, best_val, test_at_best)`.

After merging these two files on `param_count`, we obtain a cycle-level table:

```text
cycle_id  param_count  best_val   test_at_best
0         880,578      0.575695   0.631096
1         889,796      0.575655   0.644147
```

### 4.1 Parameter Counts per Cycle

Suggested plot: `plots/cycle_param_counts_seed60.png`

* **Cycle 0:** `880,578` parameters
* **Cycle 1:** `889,796` parameters

The parameter count **increases slightly** (+9,218 parameters, ≈+1 %) in cycle 1.
This indicates that, in this run, PAI tends to **expand** the model a bit during restructuring rather than prune it.

### 4.2 Cycle-Level Performance

Suggested plot: `plots/cycle_best_scores_seed60.png`

* **Cycle 0:**

  * `best_val = 0.575695`
  * `test_at_best = 0.631096`
* **Cycle 1:**

  * `best_val = 0.575655` (slightly better validation)
  * `test_at_best = 0.644147` (worse test performance)

Additional summary (from the notebook):

* **Best Val MAE across cycles (cycle-level):** `0.575655`
* **Best cycle id:** `1`

Interpretation:

* At the **cycle level**, the second cycle finds a marginally better validation optimum than the first (difference ≈ 4e-5).
* However, the **test MAE at cycle 1’s validation best** is **worse** than cycle 0’s test MAE (0.644 vs 0.631).
* The **global best test MAE (0.632334)** reported at the end of training occurs at a later epoch inside the final cycle, not exactly at the cycle-level validation minimum stored in `best_test_scores.csv`.

This mismatch between `best_val` and `test_at_best` across cycles is a good reminder that:

* PAI (and our early-stopping criterion) focuses on **validation error**, not test error.
* Slight overfitting to the validation set is possible when doing architecture search on a single split.

---

## 5. Discussion and Limitations

**Effectiveness of PAI**

* On this **seed=60** run, PAI improves both validation and test MAE by about **1.8 %** over a strong NNConv baseline.
* The improvement is modest but consistent and obtained **without changing the base architecture or loss function**.
* PAI’s structural changes are small (≈1 % more parameters) but enough to slightly push the model to a better optimum.

**Cost of PAI**

* The per-epoch runtime after the first switch is only **≈5 % slower**, and there are only a couple of switches.
* For this experiment, PAI behaves like a **lightweight architecture search** mechanism, rather than an expensive meta-learning loop.

**Generalization vs validation**

* Cycle-level results show that the cycle with the best validation MAE does **not** have the best test MAE.
* This highlights a potential risk of **overfitting to validation** when the same split is used for both hyperparameter tuning and architecture selection.

**Limitations**

* All conclusions are based on a **single random seed** and a **single dataset**.
* We did not run a systematic **multi-seed study** (e.g., 5 seeds) to estimate variance in PAI’s gains.
* Hyperparameters for PAI (switch thresholds, history length) were chosen heuristically and not tuned.
* Alternative backbones (GIN, GAT, deeper NNConv, etc.) were not explored.

**Future work**

* Run Baseline and PAI over multiple seeds and report mean ± std of MAE.
* Tune PAI switch hyperparameters more carefully and study their impact.
* Compare against other regularization and architecture-search baselines (e.g., dropout schedules, width/depth changes, Neural Architecture Search).
* Evaluate PAI on additional graph benchmarks to test robustness.

---

## 6. How to Reproduce

To reproduce this analysis:

1. **Train Baseline + PAI**

   ```bash
   # main script name depends on your repo; e.g.:
   python train_pai_qm.py
   ```

   Make sure `CFG.seed = 60` and `CFG.save_name = 'PAI'`.

2. **Inspect outputs**

   After training, check:

   ```text
   outputs/PAI_seed60/
     Scores.csv
     Times.csv
     LearningRate.csv
     paramCounts.csv
     switchEpochs.csv
     best_test_scores.csv
     pai_plots.png
   ```

3. **Run the analysis notebook**

   Open `analysis_seed60.ipynb` (or similar) and run the cells that:

   * Load the CSVs from `outputs/PAI_seed60/`.
   * Generate the plots shown in this document.
   * Print the summary statistics (best MAEs, average epoch times, cycle table).

4. **Regenerate this report**

   The figures referenced here can be saved under `plots/`:

   ```text
   plots/
     validation_curves_seed60.png
     lr_schedule_seed60.png
     epoch_time_seed60.png
     cycle_param_counts_seed60.png
     cycle_best_scores_seed60.png
   ```

   Commit both the CSVs and the plots to GitHub, together with this `analysis.md`, to provide a complete, reproducible record of the experiment.

```
::contentReference[oaicite:0]{index=0}
```
