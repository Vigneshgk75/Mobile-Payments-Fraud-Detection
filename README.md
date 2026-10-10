# Leakage-aware fraud detection pipeline

Companion code for "Beyond the Balance Leak". One script, four datasets.

## Setup
    pip install -r requirements.txt

## Smoke test (toy data, ~1 minute)
    python make_toy_data.py toy.csv 200000
    python fraud_pipeline.py --dataset paysim --path toy.csv --models xgb lgbm --boot 20 --out res

## Real runs
| Dataset  | --dataset | --path |
|----------|-----------|--------|
| PaySim   | paysim    | PS_20174392719_1491204439457_log.csv |
| BankSim  | banksim   | bs140513_032310.csv |
| IEEE-CIS | ieee_cis  | train_transaction.csv |
| ULB      | ulb       | creditcard.csv |

First run (fast, low RAM): 
    python fraud_pipeline.py --dataset paysim --path <csv> --models xgb lgbm --treatments weight --seeds 1 --boot 0 --out results

Match your earlier 80/20 split (test = last ~20% of time; use the first step of your test set):
    ... --test-from <step> --train-frac 0.8 --val-frac 0.2

Final paper run (hours on full PaySim; run on a machine with 16 GB+ RAM):
    python fraud_pipeline.py --dataset paysim --path <csv> --models lr xgb lgbm cat iforest \
        --treatments none weight undersample --seeds 5 --boot 1000 --out results

Random forest on all 5M training rows is slow and memory-hungry; use it with `--treatments undersample`.
Low RAM? Add `--train-neg-frac 0.2` (keeps all fraud, 20% of legitimate training rows; report it as a deviation).

SMOTE (`--treatments smote`) is only practical on ULB/BankSim-sized data.

## Outputs (results/<dataset>/)
- metrics.csv: every feature set x model x treatment x seed (Tables 6-8)
- summary_pr_auc.csv: F0/F1/F2 PR-AUC and the F1-F2 drop (Table 5)
- audit.json: flagged features, top-4 importance share, shuffled-label check, F2 rolling stability
- shap_groups_F1/F2.csv, shap_features_F1/F2.csv (Table 9)

## Feature sets
F0 raw only; F1 raw + balance + behavioral (pre-audit); F2 raw + behavioral without balance
columns. Add `--auto-drop` to also remove audit-flagged features from F2, but review
audit.json first: a high univariate AUC is a prompt to check the data card, not proof of leakage.

## Known limits
- Fixed hyperparameters (no Optuna yet); add tuning before final numbers.
- Elliptic (graph) and BAF are not wired in; they need their own loaders.
- IEEE-CIS entity is a proxy (card1 + addr1), so behavioral features are approximate.
- Behavioral windows use strictly earlier timestamps; same-timestamp events are excluded.
- PaySim isFlaggedFraud is ignored.

## Final runs: tuning, seeds, confidence intervals
    python fraud_pipeline.py --dataset banksim --path bs140513_032310.csv --models xgb lgbm \
        --treatments weight --tune 50 --seeds 5 --boot 1000 --out final
    python fraud_pipeline.py --dataset ulb --path creditcard.csv --models xgb lgbm \
        --treatments weight --tune 50 --seeds 5 --boot 1000 --out final
    python fraud_pipeline.py --dataset ieee_cis --path train_transaction.csv --models xgb lgbm \
        --treatments weight --tune 50 --seeds 5 --boot 1000 --out final
    python fraud_pipeline.py --dataset paysim --path PS_20174392719_1491204439457_log.csv --models xgb lgbm \
        --treatments weight --tune 30 --seeds 5 --boot 200 --out final   # slowest: run overnight

- `--tune N`: Optuna search on the validation period (maximises PR-AUC) for xgb and lgbm, using the
  deployable feature set F2 and all frauds plus a negative subsample (`--tune-neg-cap`, weights keep the
  class balance). The best parameters are reused for F0, F1 and F2 and saved in best_params.json.
- `--seeds N`: repeats every fit with different seeds; summary_with_spread.csv gives mean and std.
- `--boot B`: block-bootstrap CI for PR-AUC (seed 0 only); paired_differences.csv gives CIs for
  behavioral_gain_F2_minus_F0 and feature_gap_F1_minus_F2 using identical resamples. A CI that includes 0 means
  the difference is not distinguishable from noise.