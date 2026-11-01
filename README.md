# Credit card fraud detection (IEEE-CIS)

XGBoost on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) data, built around
one rule: **evaluate the way the model would be used** — train on the past, score the future, and look at the
held-out test months exactly once.

| Held-out test (months 5–6, scored once) | Value |
|---|---|
| PR-AUC (average precision) | **0.648** (validation promised 0.654) |
| ROC-AUC | 0.936 |
| Top 2% of traffic reviewed | 47% of frauds caught at 82% precision |
| Fraud + review cost vs. no model ($3/review, $10/false alarm) | **−67%** |
| Value of one month fresher labels | **+0.066 PR-AUC** — more than all hyperparameter tuning |

## Why the evaluation is the point

IEEE-CIS is 6 months of transactions (~590K rows, 3.5% fraud). A random split leaks the future: the dataset
labels a *client*, so once a card is reported its later transactions are fraud too, and a random split lets the
model meet the same clients on both sides. Here:

- **Time split** in 30-day months: train 0–3, validate 4, test 5–6.
- **Every feature is causal** — a row only sees earlier rows — so features can be built once on the full table.
- **Paired 3-seed comparisons.** Seed noise was ±0.004–0.017 PR-AUC, as large as many feature gains; a change
  is kept only if it wins on the same seeds.
- **Test months untouched** until every choice (features, parameters, tree count, calibration, alert policy)
  was frozen. Test feature *distributions* were inspected for drift; test *labels* were read once, at the end.

## Results on validation (month 4, mean of 3 seeds)

| Step | PR-AUC | ROC-AUC | Kept? |
|---|---|---|---|
| Baseline: raw columns | 0.5735 | 0.9140 | |
| + client features (card1 + addr1 + D1n identity, expanding stats, velocity) | 0.5913 | 0.9210 | ✓ |
| + entity frequency (24h counts, 30-day share per card/email/device/amount) | 0.5994 | 0.9224 | ✓ |
| + D-column normalisation (`day − D` for anchor columns, replacing raw) | 0.6243 | 0.9271 | ✓ |
| + local hour | 0.6361 | 0.9308 | ✓ |
| + device parsing (browser/OS family + versions behind newest) | 0.6422 | 0.9312 | ✓ |
| + tuning: max_depth 8 → 12 | **0.6540** | **0.9361** | ✓ |

Tried and rejected (paired, 3 seeds): missing-value counts (+0.0003), correlated-V pruning (−0.0105),
daily entity *rank* instead of share (−0.0013, worse under label delay), learning rate 0.1, `min_child_weight` 5
(a hessian sum — hurts on rare positives), depth 14 (+0.0002, slower).

## What else the project measured

**Explanations (SHAP).** Engineered features carry a third of the model's attribution. `device_brand` ranks #2
by SHAP but #85 by gain — mostly its "missing" value, i.e. no identity record.

**Thresholds and calibration.** A fixed score cut moved a lot between seeds; a review *share* did not.
Raw scores were overconfident and too low on average (Platt slope ≈ 0.7). Platt scaling (kept) fixed calibration
without changing the ranking; isotonic created ties and lost PR-AUC. With calibrated probabilities, alerting when
`p · amount − review cost − (1 − p) · false-alarm cost > 0` beat every fixed share.

**Drift and decay.**
- A one-month label delay cost **−15.6% PR-AUC** (train 0–2, validate 4).
- Within the validation month, weekly PR-AUC fell 0.80 → 0.55 while the mean predicted score stayed flat:
  **unlabelled score monitoring would not have noticed**.
- The decay has two parts. Clients already seen with fraud (2% of rows, 31% of frauds, PR-AUC 0.96) are
  recognised, and their share shrinks each week; and skill on new clients falls too.
- Adversarial validation separates train from test almost perfectly (AUC 0.999): normalised D columns and
  browser versions (43% of test identity rows have a browser version never seen in training).

## Final model and the one-time test

The final model is the frozen recipe trained on months 0–4 with a fixed 2,100 trees (no month left to
early-stop on), averaged over 3 seeds. Its Platt calibration and review share were fitted on month 4 using the
same recipe trained on months 0–3, so no test label shaped any choice.

| Test months 5–6 (92,427 transactions, 3,213 frauds) | PR-AUC | ROC-AUC |
|---|---|---|
| Final model (months 0–4) | **0.6482** | **0.9355** |
| Same recipe on months 0–3 (one month staler) | 0.5822 | 0.9123 |

| Alert policy on the test months | Alerts | Fraud $ caught | Total cost | Saved |
|---|---|---|---|---|
| No model | 0 | $0 | $487k | — |
| Review top 7.5% (share chosen on month 4) | 6,950 | $334k | $219k | 55% |
| **Expected $ > 0, Platt** | 6,523 | **$390k** | **$161k** | **67%** |

Calibration transferred (ECE 0.002; predicted fraud rate 3.58% vs actual 3.48%). New clients — 55% of rows,
75% of frauds — scored PR-AUC 0.594, the same as on validation.

**Takeaway:** validation was an honest estimate, and the largest lever left is retraining frequency, not model
capacity.

The Kaggle submission (`scripts/submit.py`) retrains the same recipe on all six months. The competition's test
period starts ~30 days after the last label, so it measures the label-delay scenario on ROC-AUC only.
<!-- Kaggle leaderboard: public ROC-AUC _____ / private _____ -->

## Repository

```
src/fraud/
  data/        load, merge and downcast the CSVs; dev sample; Kaggle test loader
  features/    causal feature steps (client, entity, dnorm, clock, device, ...) and the pipeline
  split.py     time split and robustness scenarios (valid3, gap, recent3)
  baseline.py  XGBoost parameters and training with early stopping on PR-AUC
  metrics.py   ROC-AUC, PR-AUC, lift
  explain.py   SHAP helpers
  threshold.py capacity table and cost curve
  calibration.py  Platt / isotonic, reliability, expected-benefit rule
  drift.py     PSI, adversarial validation, weekly and per-segment decay
  final.py     fixed-tree seed ensemble
scripts/       one script per phase (run_experiment, explain, threshold, calibrate, drift, decay, final, submit)
tests/         pytest suite (182 tests) on small synthetic data
```

## Running it

Locally (Python 3.13, [uv](https://docs.astral.sh/uv/)), on a 50K-row sample:

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/make_sample.py
uv run python scripts/build_dataset.py --source sample
```

Full runs used a Kaggle notebook with a GPU and the competition data attached:

```bash
git clone https://github.com/d-labsystem/ieee-cis-fraud-detection repo
PYTHONPATH=repo/src python repo/scripts/build_dataset.py --source raw
PYTHONPATH=repo/src python repo/scripts/run_experiment.py --device cuda --steps client entity dnorm_replace hour device
PYTHONPATH=repo/src python repo/scripts/final.py --path /kaggle/working/train_raw.parquet --device cuda
PYTHONPATH=repo/src python repo/scripts/submit.py --device cuda
```

Each script's docstring explains its phase and arguments.
