"""Numerical definitions shared by the manuscript analysis entry points."""
from pathlib import Path
import hashlib
import json
import platform
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd
from scipy import linalg, stats

ROOT = Path(__file__).resolve().parents[3]
SEED = 20260822
KEY = ["state_alpha", "year", "month"]
DISPLAY_CLASSES = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = np.std(x, ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("An exposure with zero/nonfinite variance is not estimable")
    return (x - x.mean()) / sd


def adjust_p(values, method="BY", family_size=None):
    """Retain the declared testing family, including non-estimable tests."""
    p = np.asarray(values, float)
    ok = np.isfinite(p)
    m = len(p) if family_size is None else family_size
    if m < ok.sum():
        raise ValueError("Testing family smaller than number of available P values")
    out = np.full(len(p), np.nan)
    if not ok.any():
        return out
    order = np.argsort(p[ok], kind="stable")
    factor = np.sum(1 / np.arange(1, m + 1)) if method == "BY" else 1
    ranked = p[ok][order] * m * factor / np.arange(1, ok.sum() + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    restored = np.empty(ok.sum())
    restored[order] = q
    out[ok] = restored
    return out


def fe_matrix(d, specification="year_month"):
    columns = [np.ones((len(d), 1))]
    groups = [d.state_alpha]
    if specification == "year_month":
        groups += [d.year.astype(str) + "-" + d.month.astype(str)]
    elif specification == "year_and_month":
        groups += [d.year, d.month]
    elif specification == "within_state":
        groups = [d.year, d.month]
    elif specification == "state_trend":
        groups += [d.month]
        states = pd.get_dummies(d.state_alpha, dtype=float).to_numpy()
        columns += [states * (d.year.to_numpy(float) - d.year.mean())[:, None]]
    else:
        raise ValueError(specification)
    columns += [pd.get_dummies(g.astype(str), drop_first=True, dtype=float).to_numpy() for g in groups]
    return np.column_stack(columns)


def wls(y, x, weights):
    sw = np.sqrt(weights / np.mean(weights))
    coef, _, rank, _ = linalg.lstsq(x * sw[:, None], y * sw, lapack_driver="gelsd")
    return coef, y - x @ coef, int(rank)


def r_squared(y, residual, w, rank):
    n = len(y)
    sst = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    sse = float(np.sum(w * residual ** 2))
    r2 = 1 - sse / sst if sst > 0 else np.nan
    adjusted = 1 - (1-r2) * (n-1)/(n-rank) if n > rank else np.nan
    return r2, adjusted


def fit_panel(d, outcome, exposure, specification="year_month", weighted=True, standardize_outcome=False):
    required = list(dict.fromkeys(KEY + [outcome, exposure, "milk_cows_head"]))
    d = d[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = d[(d.milk_cows_head > 0) & (d[outcome] > 0)]
    result = {"n": len(d), "n_states": d.state_alpha.nunique(), "status": "too_few"}
    min_states = 1 if specification == "within_state" else 3
    if len(d) < 60 or d.state_alpha.nunique() < min_states:
        return result
    if d[exposure].nunique() < 2:
        return dict(result, status="constant_exposure")
    y = np.log(d[outcome].to_numpy(float))
    if standardize_outcome:
        y = zscore(y)
    x = zscore(d[exposure].to_numpy(float))
    w = d.milk_cows_head.to_numpy(float) if weighted else np.ones(len(d))
    fe = fe_matrix(d, specification)
    design = np.column_stack([fe, x])
    coefficient, residual, rank = wls(y, design, w)
    _, base_residual, rank_base = wls(y, fe, w)
    if rank != rank_base + 1:
        return dict(result, status="collinear")
    n, g = len(d), d.state_alpha.nunique()
    if n <= rank:
        return dict(result, status="no_residual_df")
    sw = np.sqrt(w / w.mean())
    xw = design * sw[:, None]
    rw = residual * sw
    bread = np.linalg.pinv(xw.T @ xw)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for state in d.state_alpha.unique():
        mask = d.state_alpha.eq(state).to_numpy()
        score = xw[mask].T @ rw[mask]
        meat += np.outer(score, score)
    if g > 1:
        cov = bread @ meat @ bread * g/(g-1) * (n-1)/(n-rank)
        se = float(np.sqrt(max(cov[-1, -1], 0)))
        df = g-1
    else:
        # State-specific priority models do not report clustered significance.
        se, df = np.nan, n-rank
    beta = float(coefficient[-1])
    p = float(2*stats.t.sf(abs(beta/se), df)) if se > 0 else np.nan
    critical = stats.t.ppf(.975, df)
    full_r2, full_adj = r_squared(y, residual, w, rank)
    base_r2, base_adj = r_squared(y, base_residual, w, rank_base)
    return dict(result, status="ok", beta=beta, se=se, p=p,
                ci_low=beta-critical*se, ci_high=beta+critical*se,
                r2=full_r2, adjusted_r2=full_adj, baseline_r2=base_r2,
                incremental_r2=full_r2-base_r2,
                adjusted_incremental_r2=full_adj-base_adj,
                outcome_scale="z(log outcome)" if standardize_outcome else "log outcome")


def write_environment(output, inputs=()):
    versions = {"Python": platform.python_version()}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "shap", "openpyxl"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    sources = list(Path(__file__).parent.rglob("*.py")) + list(Path(__file__).parent.rglob("*.R"))
    hashes = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in [*inputs, *sources]}
    Path(output).write_text(json.dumps({"versions": versions, "seed": SEED, "sha256": hashes}, indent=2))
