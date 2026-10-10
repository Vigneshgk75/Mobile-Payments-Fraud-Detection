#!/usr/bin/env python3
"""Leakage-aware fraud detection pipeline.

Implements the protocol of the paper "Beyond the Balance Leak":
  - strictly causal behavioral features (windows end BEFORE the current event)
  - feature sets F0 (raw), F1 (full, pre-audit), F2 (deployable)
  - leakage audit (univariate AUC screen, importance concentration, shuffled labels)
  - chronological 60/15/25 split, validation-tuned threshold (max F2)
  - models: lr, rf, xgb, lgbm, cat, iforest; treatments: none, weight, undersample, smote
  - metrics: PR-AUC (+ block-bootstrap CI), ROC-AUC, P/R/F1/MCC, precision@k,
    recall@FPR, expected cost, Brier, ECE
  - SHAP group importance and rolling-origin importance stability

Usage:
  python fraud_pipeline.py --dataset paysim --path PS_20174392719_1491204439457_log.csv \
      --models xgb lgbm --treatments weight --out results
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Dataset configuration. Windows are in the dataset's native time unit.
# "bal" columns are the known post-event state variables (audited, not deployable).
# --------------------------------------------------------------------------
CFG = {
    "paysim": dict(
        time="step", label="isFraud", amount="amount", sender="nameOrig", receiver="nameDest",
        cat=["type"], windows=[1, 24, 168], hour=lambda t: t % 24, dow=lambda t: (t // 24) % 7,
        bal=["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"],
        ignore=["isFlaggedFraud"]),
    "banksim": dict(
        time="step", label="fraud", amount="amount", sender="customer", receiver="merchant",
        cat=["category", "age", "gender"], windows=[1, 7, 30], hour=None, dow=lambda t: t % 7,
        bal=[], ignore=["zipcodeOri", "zipMerchant"]),
    "ieee_cis": dict(  # pass train_transaction.csv; sender is a proxy uid (card1 + addr1)
        time="TransactionDT", label="isFraud", amount="TransactionAmt", sender="uid", receiver=None,
        cat=["ProductCD", "card4", "card6"], windows=[3600, 86400, 604800],
        hour=lambda t: (t // 3600) % 24, dow=lambda t: (t // 86400) % 7,
        bal=[], ignore=["TransactionID"], numeric="auto"),
    "ulb": dict(  # creditcard.csv: PCA components only, no entities, so no behavioral features
        time="Time", label="Class", amount="Amount", sender=None, receiver=None, cat=[],
        windows=[], hour=lambda t: (t // 3600) % 24, dow=None, bal=[], ignore=[], numeric="auto"),
}


def load(name, path):
    c = CFG[name]
    head = pd.read_csv(path, nrows=0).columns
    cats = {col: "category" for col in c["cat"] + [c["sender"], c["receiver"]] if col in head}
    df = pd.read_csv(path, dtype=cats)  # string columns become categories: far less RAM
    for col in df.select_dtypes("float64").columns:
        if col != c["amount"]:
            df[col] = df[col].astype("float32")
    if name == "ieee_cis":
        df["uid"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
    df = df.sort_values(c["time"], kind="stable").reset_index(drop=True)
    return df, c


# --------------------------------------------------------------------------
# Causal entity features. For each event, only events of the same entity with a
# strictly smaller timestamp are used (same-time events and the event itself excluded).
# --------------------------------------------------------------------------
def entity_feats(t, ent, amt, windows, prefix):
    t = t - t.min()
    codes = pd.factorize(ent)[0].astype(np.int64)
    span = int(t.max()) + int(max(windows)) + 2
    key = codes * span + t
    order = np.argsort(key, kind="stable")
    ks, a, ts, cs_codes = key[order], amt[order], t[order], codes[order]
    csum = np.concatenate([[0.0], np.cumsum(a)])
    csq = np.concatenate([[0.0], np.cumsum(a * a)])
    start = np.searchsorted(ks, cs_codes * span, "left")
    hi = np.searchsorted(ks, ks, "left")  # first event with the same key: all before it are strictly earlier
    out = {}
    n = hi - start
    out[f"{prefix}_hist_n"] = n
    mean = np.where(n > 0, (csum[hi] - csum[start]) / np.maximum(n, 1), np.nan)
    var = np.where(n > 1, (csq[hi] - csq[start]) / np.maximum(n, 1) - mean ** 2, np.nan)
    std = np.sqrt(np.maximum(var, 0))
    out[f"{prefix}_hist_mean"] = mean
    out[f"{prefix}_amt_ratio"] = np.where(n > 0, a / np.maximum(mean, 1e-9), np.nan)
    out[f"{prefix}_amt_z"] = np.where(std > 0, (a - mean) / np.maximum(std, 1e-9), np.nan)
    out[f"{prefix}_since_last"] = np.where(n > 0, ts - ts[np.maximum(hi - 1, 0)], np.nan)
    for w in windows:
        lo = np.searchsorted(ks, np.maximum(ks - w, cs_codes * span), "left")
        out[f"{prefix}_cnt_{w}"] = hi - lo
        out[f"{prefix}_sum_{w}"] = csum[hi] - csum[lo]
    res = {}
    for k, v in out.items():
        arr = np.empty(len(v), dtype=np.float32)
        arr[order] = v
        res[k] = arr
    return res


def build_features(df, c):
    t = df[c["time"]].to_numpy().astype(np.int64)
    amt = df[c["amount"]].to_numpy(float)
    X = {"raw_amount": amt, "raw_log_amount": np.log1p(np.maximum(amt, 0))}
    if c.get("hour"):
        h = c["hour"](t)
        X["raw_hour_sin"], X["raw_hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    if c.get("dow"):
        d = c["dow"](t)
        X["raw_dow_sin"], X["raw_dow_cos"] = np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)
    for col in c["cat"]:
        X["raw_" + col] = df[col].astype("category").cat.codes.to_numpy()
    skip = {c["time"], c["label"], c["amount"], c["sender"], c["receiver"], *c["cat"], *c["bal"], *c["ignore"]}
    if c.get("numeric") == "auto":
        for col in df.select_dtypes("number").columns:
            if col not in skip and col not in ("uid",):
                X["raw_" + col] = df[col].to_numpy()
    for col in c["bal"]:
        X["bal_" + col] = df[col].to_numpy(float)
    if len(c["bal"]) == 4:
        o0, o1, d0, d1 = (df[x].to_numpy(float) for x in c["bal"])
        X["bal_err_orig"], X["bal_err_dest"] = o1 + amt - o0, d0 + amt - d1
    W = c["windows"]
    if c["sender"] and W:
        sf = entity_feats(t, df[c["sender"]], amt, W, "snd")
        X.update(sf)
        X["vel_burst"] = sf[f"snd_cnt_{W[0]}"] / (sf[f"snd_cnt_{W[-1]}"] * W[0] / W[-1] + 1)
    if c["receiver"] and W:
        X.update(entity_feats(t, df[c["receiver"]], amt, W, "rcv"))
        sc, rc = pd.factorize(df[c["sender"]])[0].astype(np.int64), pd.factorize(df[c["receiver"]])[0]
        pf = entity_feats(t, sc * (int(rc.max()) + 1) + rc, amt, [W[-1]], "rel")
        X["rel_pair_n"], X["rel_first_pair"] = pf["rel_hist_n"], (pf["rel_hist_n"] == 0).astype(np.float32)
        X["rel_pair_amt_ratio"] = pf["rel_amt_ratio"]
    # fill a preallocated float32 matrix column by column, freeing each array as we go
    names = list(X)
    M = np.empty((len(t), len(names)), dtype=np.float32)
    for j, k in enumerate(names):
        col = np.array(X.pop(k), dtype=np.float32)  # copy: source arrays can be read-only views
        col[~np.isfinite(col)] = -1.0  # NaN/inf (no history, divide by zero) -> -1
        M[:, j] = col
    return pd.DataFrame(M, columns=names, copy=False)


def split_idx(t, train_frac=0.60, val_frac=0.15, test_from=None):
    """Rows are time-sorted. Default: first 60% train, next 15% validation, rest test.
    With test_from=T the test set is every row with time >= T (e.g. to match an earlier 80/20 split);
    the rows before T are divided into train/validation in the ratio train_frac : val_frac."""
    n = len(t)
    if test_from is not None:
        i2 = int(np.searchsorted(t, test_from, "left"))
        i1 = int(i2 * train_frac / (train_frac + val_frac))
    else:
        i1, i2 = int(train_frac * n), int((train_frac + val_frac) * n)
    return np.arange(0, i1), np.arange(i1, i2), np.arange(i2, n)


def group_of(col):
    return col.split("_")[0]


# --------------------------------------------------------------------------
# Models and treatments
# --------------------------------------------------------------------------
def make_model(name, seed, params=None):
    p = dict(params or {})
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier
        base = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)
        base.update(p)
        return XGBClassifier(**base, tree_method="hist", eval_metric="aucpr", random_state=seed, n_jobs=-1)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        base = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8)
        base.update(p)
        return LGBMClassifier(**base, subsample_freq=1, random_state=seed, verbose=-1)
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=6, verbose=0, random_seed=seed)
    if name == "iforest":
        return IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    raise ValueError(name)


def apply_treatment(X, y, treat, seed):
    rng = np.random.default_rng(seed)
    if treat == "undersample":
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        keep = np.concatenate([pos, rng.choice(neg, min(len(neg), 10 * len(pos)), replace=False)])
        return X.iloc[keep], y[keep], None
    if treat == "smote":  # practical only on small datasets; subsample negatives first for large ones
        from imblearn.over_sampling import SMOTE
        Xr, yr = SMOTE(sampling_strategy=0.1, random_state=seed).fit_resample(X, y)
        return Xr, yr, None
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0) if treat == "weight" else None
    return X, y, w


def fit_predict(name, Xtr, ytr, treat, seed, *Xs, params=None):
    model = make_model(name, seed, params)
    if name == "iforest":
        model.fit(Xtr)
        return model, [-model.score_samples(x) for x in Xs]
    Xt, yt, w = apply_treatment(Xtr, ytr, treat, seed)
    if name == "lr":
        model.fit(Xt, yt, logisticregression__sample_weight=w)
    else:
        model.fit(Xt, yt, sample_weight=w)
    return model, [model.predict_proba(x)[:, 1] for x in Xs]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def best_f2_threshold(y, s):
    p, r, thr = precision_recall_curve(y, s)
    f2 = 5 * p[:-1] * r[:-1] / np.maximum(4 * p[:-1] + r[:-1], 1e-12)
    return float(thr[np.argmax(f2)]) if len(thr) else 0.5


def ece(y, s, bins=10):
    idx = np.minimum((s * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - s[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def block_bootstrap_ap(y, s, n_boot, blocks=50, seed=0):
    if n_boot <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    parts = np.array_split(np.arange(len(y)), blocks)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([parts[i] for i in rng.integers(0, blocks, blocks)])
        if y[idx].sum() > 0:
            vals.append(average_precision_score(y[idx], s[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def evaluate(y, s, amt, thr_val, probabilistic, n_boot):
    out = {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s)}
    lo, hi = block_bootstrap_ap(y, s, n_boot)
    out["pr_auc_lo"], out["pr_auc_hi"] = lo, hi
    if probabilistic:
        out["brier"], out["ece"] = brier_score_loss(y, s), ece(y, s)
    for tag, thr in (("t05", 0.5), ("tval", thr_val)):
        if tag == "t05" and not probabilistic:
            continue
        pred = s >= thr
        out.update({f"precision_{tag}": precision_score(y, pred, zero_division=0),
                    f"recall_{tag}": recall_score(y, pred), f"f1_{tag}": f1_score(y, pred),
                    f"mcc_{tag}": matthews_corrcoef(y, pred),
                    f"fp_{tag}": int(((pred) & (y == 0)).sum()), f"tp_{tag}": int((pred & (y == 1)).sum())})
    order = np.argsort(-s)
    for k in (0.001, 0.005, 0.01):
        out[f"prec_at_{k}"] = float(y[order[:max(int(k * len(y)), 1)]].mean())
    fpr, tpr, _ = roc_curve(y, s)
    out["recall_at_fpr_0.001"] = float(np.interp(0.001, fpr, tpr))
    pred = s >= thr_val
    base = amt[y == 1].sum()
    for r in (10, 50, 200):
        c_inv = amt[y == 1].mean() / r
        cost = amt[(~pred) & (y == 1)].sum() + c_inv * ((pred) & (y == 0)).sum()
        out[f"cost_norm_1:{r}"] = float(cost / max(base, 1e-9))  # 1.0 = no model
    out["n_test"], out["n_fraud_test"] = len(y), int(y.sum())
    return out


def tune(name, Xtr, ytr, Xva, yva, n_trials, seed, neg_cap=300_000):
    """Optuna search maximising validation PR-AUC. Trains on all frauds plus a negative subsample whose
    weights (1/f) keep the class-weighting of the full training set; validation is the full validation period."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    keep = rng.choice(neg, min(len(neg), neg_cap), replace=False)
    idx = np.sort(np.concatenate([pos, keep]))
    Xs, ys = Xtr.iloc[idx], ytr[idx]
    f = len(keep) / len(neg)
    w = np.where(ys == 1, len(neg) / max(len(pos), 1), 1.0 / f)

    def objective(trial):
        if name == "xgb":
            p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 500),
                     max_depth=trial.suggest_int("max_depth", 3, 10),
                     learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                     min_child_weight=trial.suggest_float("min_child_weight", 0.1, 10, log=True),
                     reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10, log=True))
        else:
            p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 500),
                     num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                     learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                     min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                     reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10, log=True))
        m = make_model(name, seed, p)
        m.fit(Xs, ys, sample_weight=w)
        return average_precision_score(yva, m.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


def paired_bootstrap(y, sa, sb, n_boot, blocks=50, seed=0):
    """Block-bootstrap CI for PR-AUC(sb) - PR-AUC(sa), using identical resamples for both score vectors."""
    rng = np.random.default_rng(seed)
    parts = np.array_split(np.arange(len(y)), blocks)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([parts[i] for i in rng.integers(0, blocks, blocks)])
        if y[idx].sum() > 0:
            d.append(average_precision_score(y[idx], sb[idx]) - average_precision_score(y[idx], sa[idx]))
    point = float(average_precision_score(y, sb) - average_precision_score(y, sa))
    lo, hi = np.percentile(d, [2.5, 97.5]) if d else (np.nan, np.nan)
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------
# Audit, SHAP, rolling stability
# --------------------------------------------------------------------------
def audit(Xtr, ytr, seed, n_boot_dummy=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    sub = np.concatenate([pos, rng.choice(neg, min(len(neg), 300_000), replace=False)])
    Xs, ys = Xtr.iloc[sub], ytr[sub]
    uni = {}
    for col in Xs.columns:
        v = Xs[col].to_numpy(float)
        v = np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(v).any() else 0.0)
        uni[col] = float(roc_auc_score(ys, v)) if np.ptp(v) > 0 else 0.5
    flagged = {k: v for k, v in uni.items() if v > 0.95 or v < 0.05}
    model, _ = fit_predict("xgb", Xtr, ytr, "weight", seed)
    imp = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    return {"univariate_auc": uni, "flagged": flagged,
            "top4_share": float(imp.head(4).sum() / max(imp.sum(), 1e-12)),
            "top4": imp.head(4).to_dict(), "importance": imp.to_dict()}


def shuffled_label_check(Xtr, ytr, Xte, yte, seed):
    rng = np.random.default_rng(seed)
    _, (s,) = fit_predict("xgb", Xtr, rng.permutation(ytr), "weight", seed, Xte)
    return {"pr_auc_shuffled": float(average_precision_score(yte, s)), "prevalence_test": float(yte.mean())}


def shap_groups(model, Xte, yte, seed, n=20000):
    import shap
    rng = np.random.default_rng(seed)
    pos, neg = np.where(yte == 1)[0], np.where(yte == 0)[0]
    idx = np.concatenate([rng.choice(pos, min(len(pos), n // 4), replace=False),
                          rng.choice(neg, min(len(neg), n * 3 // 4), replace=False)])
    sv = shap.TreeExplainer(model).shap_values(Xte.iloc[idx])
    sv = sv[1] if isinstance(sv, list) else sv
    m = pd.Series(np.abs(sv).mean(0), index=Xte.columns)
    return m.sort_values(ascending=False), m.groupby(group_of).sum().sort_values(ascending=False)


def rolling_stability(X, y, cols, seed, k=5):
    n = len(y)
    cut = np.linspace(0.4, 0.9, k)
    imps, aps = [], []
    for i, f in enumerate(cut):
        a, b = int(f * n), int(min(f + 0.1, 1.0) * n)
        model, (s,) = fit_predict("xgb", X.iloc[:a][cols], y[:a], "weight", seed, X.iloc[a:b][cols])
        imps.append(pd.Series(model.feature_importances_, index=cols))
        aps.append(average_precision_score(y[a:b], s) if y[a:b].sum() else np.nan)
    rho = [spearmanr(imps[i], imps[i + 1])[0] for i in range(len(imps) - 1)]
    return {"pr_auc_by_window": aps, "spearman_consecutive": rho, "spearman_mean": float(np.nanmean(rho))}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG))
    ap.add_argument("--path", required=True)
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "xgb", "lgbm"])
    ap.add_argument("--treatments", nargs="+", default=["weight"],
                    choices=["none", "weight", "undersample", "smote"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--boot", type=int, default=200, help="block-bootstrap resamples, run for seed 0 only (use 1000 for the paper)")
    ap.add_argument("--tune", type=int, default=0, help="Optuna trials per model (xgb, lgbm); 0 = fixed defaults")
    ap.add_argument("--tune-neg-cap", type=int, default=300000, help="negatives used while tuning (all frauds kept)")
    ap.add_argument("--auto-drop", action="store_true",
                    help="also drop audit-flagged features from F2 (review the audit first)")
    ap.add_argument("--rolling", type=int, default=5, help="rolling-origin windows for stability (0 or 1 = skip)")
    ap.add_argument("--sets", nargs="+", default=["F0", "F1", "F2"], choices=["F0", "F1", "F2"],
                    help="feature sets to run; identical sets are fitted once and reused")
    ap.add_argument("--skip-audit", action="store_true", help="skip the audit fits (reuse an earlier audit.json)")
    ap.add_argument("--no-shap", action="store_true", help="skip the SHAP step")
    ap.add_argument("--train-frac", type=float, default=0.60)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-from", type=float, default=None,
                    help="time value where the test period starts (overrides the fractions for the test set)")
    ap.add_argument("--train-neg-frac", type=float, default=1.0,
                    help="keep this fraction of legitimate TRAIN rows (low-RAM option; deviates from protocol)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = Path(a.out) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    df, c = load(a.dataset, a.path)
    y_all = df[c["label"]].to_numpy().astype(int)
    amt_all = df[c["amount"]].to_numpy(float)
    t_all = df[c["time"]].to_numpy()
    X = build_features(df, c)
    n_rows = len(df)
    del df
    tr, va, te = split_idx(t_all, a.train_frac, a.val_frac, a.test_from)
    if a.train_neg_frac < 1.0:
        rng = np.random.default_rng(0)
        keep = (y_all[tr] == 1) | (rng.random(len(tr)) < a.train_neg_frac)
        tr = tr[keep]
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr, yva, yte, amte = y_all[tr], y_all[va], y_all[te], amt_all[te]
    print(f"{a.dataset}: n={n_rows:,} fraud={y_all.sum():,} ({y_all.mean():.4%}) features={X.shape[1]}")
    print(f"train={len(tr):,} ({ytr.sum():,} fraud)  val={len(va):,} ({yva.sum():,})  test={len(te):,} ({yte.sum():,})")

    F1 = list(X.columns)
    F0 = [col for col in F1 if group_of(col) == "raw"]
    if a.skip_audit:
        rep = {"flagged": {}, "top4_share": float("nan"), "pr_auc_shuffled": float("nan"),
               "prevalence_test": float(yte.mean())}
    else:
        rep = audit(Xtr, ytr, 0)
        rep.update(shuffled_label_check(Xtr, ytr, Xte, yte, 0))
    drop = {col for col in F1 if group_of(col) == "bal"}
    if a.auto_drop:
        drop |= set(rep["flagged"])
    F2 = [col for col in F1 if col not in drop]
    rep["feature_sets"] = {"F0": len(F0), "F1": len(F1), "F2": len(F2), "dropped_in_F2": sorted(drop)}
    print(f"audit: flagged={list(rep['flagged'])} top4_share={rep['top4_share']:.3f} "
          f"shuffled PR-AUC={rep['pr_auc_shuffled']:.4f} vs prevalence={rep['prevalence_test']:.4f}")

    rows = []
    sets = {"F0": F0, "F1": F1, "F2": F2}
    best = {}
    if a.tune > 0:  # tune on the deployable set F2, reuse the parameters for F0 and F1
        for m in a.models:
            if m in ("xgb", "lgbm"):
                best[m], val_ap = tune(m, Xtr[F2], ytr, Xva[F2], yva, a.tune, 0, a.tune_neg_cap)
                print(f"tuned {m}: validation PR-AUC={val_ap:.4f} params={best[m]}")
        json.dump(best, open(out / "best_params.json", "w"), indent=2)
    scores = {}
    for seed in range(a.seeds):
        seen = {}
        for fs in a.sets:
            cols = sets[fs]
            if fs == "F0" and not F0:
                continue
            if tuple(cols) in seen:  # identical feature set already fitted: reuse its results
                src = seen[tuple(cols)]
                for r in [r for r in rows if r["feature_set"] == src and r["seed"] == seed]:
                    rows.append({**r, "feature_set": fs})
                if seed == 0:
                    for (f2, m2, t2), v in list(scores.items()):
                        if f2 == src:
                            scores[(fs, m2, t2)] = v
                continue
            seen[tuple(cols)] = fs
            for m in a.models:
                for tr_ in (["none"] if m == "iforest" else a.treatments):
                    model, (sv, st) = fit_predict(m, Xtr[cols], ytr, tr_, seed, Xva[cols], Xte[cols], params=best.get(m))
                    if m == "iforest":  # map anomaly scores to [0,1] using the validation range
                        lo, hi = sv.min(), sv.max()
                        sv, st = (sv - lo) / (hi - lo + 1e-12), np.clip((st - lo) / (hi - lo + 1e-12), 0, 1)
                    row = dict(dataset=a.dataset, feature_set=fs, model=m, treatment=tr_, seed=seed,
                               **evaluate(yte, st, amte, best_f2_threshold(yva, sv), m != "iforest", a.boot if seed == 0 else 0))
                    if seed == 0:
                        scores[(fs, m, tr_)] = st
                    rows.append(row)
                    print(f"  {fs} {m:7s} {tr_:11s} seed={seed} PR-AUC={row['pr_auc']:.4f} "
                          f"P@val={row.get('precision_tval', float('nan')):.3f} R@val={row.get('recall_tval', float('nan')):.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.csv", index=False)

    # F1 vs F2 PR-AUC drop (RQ1) for the best model family present
    summ = res.groupby(["feature_set", "model", "treatment"])["pr_auc"].mean().unstack("feature_set")
    if {"F1", "F2"} <= set(summ.columns):
        summ["drop_F1_minus_F2"] = summ["F1"] - summ["F2"]
    summ.to_csv(out / "summary_pr_auc.csv")
    print(summ.round(4))
    key = ["feature_set", "model", "treatment"]
    long = res.groupby(key)["pr_auc"].agg(pr_auc_mean="mean", pr_auc_std="std", n_seeds="count")
    if a.boot > 0:
        long = long.join(res[res.seed == 0].set_index(key)[["pr_auc_lo", "pr_auc_hi"]])
    long.to_csv(out / "summary_with_spread.csv")
    if a.boot > 0:  # paired differences: behavioral gain (F2 - F0) and leakage cost (F1 - F2)
        diffs = []
        for (fs, m, tr_), s_a in list(scores.items()):
            for lo_set, hi_set, label in (("F0", "F2", "behavioral_gain_F2_minus_F0"), ("F2", "F1", "feature_gap_F1_minus_F2")):
                if fs == lo_set and (hi_set, m, tr_) in scores and lo_set != hi_set and sets[lo_set] != sets[hi_set]:
                    d, lo, hi = paired_bootstrap(yte, s_a, scores[(hi_set, m, tr_)], a.boot)
                    diffs.append(dict(comparison=label, model=m, treatment=tr_, diff=d, ci_lo=lo, ci_hi=hi))
        pd.DataFrame(diffs).to_csv(out / "paired_differences.csv", index=False)
        print(pd.DataFrame(diffs).round(4).to_string(index=False))

    # SHAP group importance and rolling stability (xgb)
    shap_done = {}
    for fs in (() if a.no_shap else ("F1", "F2")):
        if tuple(sets[fs]) in shap_done:
            feat, grp = shap_done[tuple(sets[fs])]
        else:
            model, _ = fit_predict("xgb", Xtr[sets[fs]], ytr, "weight", 0, params=best.get("xgb"))
            feat, grp = shap_groups(model, Xte[sets[fs]], yte, 0)
            shap_done[tuple(sets[fs])] = (feat, grp)
        feat.head(30).to_csv(out / f"shap_features_{fs}.csv", header=["mean_abs_shap"])
        grp.to_csv(out / f"shap_groups_{fs}.csv", header=["mean_abs_shap"])
        rep[f"shap_top4_share_{fs}"] = float(feat.head(4).sum() / feat.sum())
        print(f"SHAP groups {fs}:", grp.round(4).to_dict())
    if a.rolling > 1:
        rep["rolling_F2"] = rolling_stability(X, y_all, F2, 0, a.rolling)
        print("rolling stability (F2):", rep["rolling_F2"])

    rep.pop("importance", None)  # univariate AUCs kept for manual review
    json.dump(rep, open(out / "audit.json", "w"), indent=2, default=float)
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()