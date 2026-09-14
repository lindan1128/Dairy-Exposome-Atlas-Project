"""Pooled exposure-by-year models and class-level summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg, stats
from common import KEY, zscore, fe_matrix, wls, fit_panel, adjust_p
from prepare import load


def fit_yearly(panel, exposure):
    d = panel[KEY+["milk_per_cow_lb", "milk_cows_head", exposure]].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d.milk_per_cow_lb > 0)&(d.milk_cows_head > 0)]
    if len(d) < 60 or d.state_alpha.nunique() < 3 or d[exposure].nunique() < 2:
        return [{"status": "too_few_or_constant", "n": len(d)}]
    y = np.log(d.milk_per_cow_lb.to_numpy(float))
    x = zscore(d[exposure].to_numpy(float))
    years = sorted(d.year.unique())
    terms = np.column_stack([x*d.year.eq(year).to_numpy() for year in years])
    w = d.milk_cows_head.to_numpy(float)
    sw = np.sqrt(w/w.mean())
    fe = fe_matrix(d, "year_and_month")
    few = fe*sw[:, None]
    q, _, _ = linalg.qr(few, mode="economic", pivoting=True)
    rank_fe = np.linalg.matrix_rank(few)
    q = q[:, :rank_fe]
    yw = y*sw
    yr = yw-q@(q.T@yw)
    xw = terms*sw[:, None]
    xr = xw-q@(q.T@xw)
    active = np.linalg.norm(xr, axis=0)>1e-9
    a = xr[:, active]
    years_active = np.asarray(years)[active]
    if not len(years_active):
        return [{"status": "collinear", "n": len(d)}]
    if np.linalg.matrix_rank(a) != a.shape[1]:
        return [{"status": "jointly_collinear_year_terms", "n": len(d)}]
    beta, _, rank_x, _ = linalg.lstsq(a, yr)
    residual = yr-a@beta
    n, g, k = len(d), d.state_alpha.nunique(), rank_fe+int(rank_x)
    if n <= k:
        return [{"status": "no_residual_df", "n": n}]
    bread = np.linalg.pinv(a.T@a)
    meat = np.zeros((a.shape[1], a.shape[1]))
    for state in d.state_alpha.unique():
        mask = d.state_alpha.eq(state).to_numpy()
        score = a[mask].T@residual[mask]
        meat += np.outer(score, score)
    cov = bread@meat@bread*g/(g-1)*(n-1)/(n-k)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    sst = np.sum((sw*(y-np.average(y, weights=w)))**2)
    sse = residual@residual
    full_r2 = 1-sse/sst
    full_adj = 1-(1-full_r2)*(n-1)/(n-k)
    rows = []
    for i, year in enumerate(years_active):
        reduced = np.delete(a, i, axis=1)
        br, _, rr, _ = linalg.lstsq(reduced, yr)
        er = yr-reduced@br
        rss = er@er
        kr = rank_fe+int(rr)
        reduced_r2 = 1-rss/sst
        reduced_adj = 1-(1-reduced_r2)*(n-1)/(n-kr)
        critical = stats.t.ppf(.975, g-1)
        rows.append({"year": int(year), "status": "ok", "n": n, "n_states": g,
                     "beta": beta[i], "se": se[i],
                     "p": 2*stats.t.sf(abs(beta[i]/se[i]), g-1) if se[i]>0 else np.nan,
                     "ci_low": beta[i]-critical*se[i], "ci_high": beta[i]+critical*se[i],
                     "r2": full_r2, "adjusted_r2": full_adj,
                     "incremental_r2": full_r2-reduced_r2,
                     "adjusted_incremental_r2": full_adj-reduced_adj,
                     "partial_r2": (rss-sse)/rss if rss>0 else np.nan})
    return rows


def class_stage_tests(yearly):
    d = yearly[yearly.status.eq("ok") & yearly.year.between(2000, 2024)].copy()
    d["abs_beta"] = d.beta.abs()
    d["stage"] = np.where(d.year<2015, "early", "late")
    rows = []
    for metric in ["abs_beta", "adjusted_incremental_r2", "partial_r2"]:
        values = d.groupby(["class_label", "exposure", "stage"])[metric].mean().unstack("stage")
        if not {"early", "late"}.issubset(values.columns):
            continue
        for cls, g in values.dropna(subset=["early", "late"]).groupby(level="class_label"):
            delta = g.late-g.early
            p = 1.0 if np.all(delta==0) else stats.wilcoxon(g.late, g.early, alternative="two-sided").pvalue
            rows.append({"class_label": cls, "metric": metric, "n_pairs": len(g), "p": p,
                         "median_early": g.early.median(), "median_late": g.late.median()})
    result = pd.DataFrame(rows)
    if not result.empty:
        for _, idx in result.groupby("metric").groups.items():
            result.loc[idx, "q_bh"] = adjust_p(result.loc[idx, "p"], "BH")
    return result


def run(output):
    output = Path(output)
    panel, meta = load(output)
    meta = meta[~meta.acute_event].copy()
    rows, recent = [], []
    for item in meta.to_dict("records"):
        exposure = item["exposure"]
        for result in fit_yearly(panel, exposure):
            rows.append(dict(item, **result))
        recent.append(dict(item, **fit_panel(panel[panel.year.between(2015, 2024)], "milk_per_cow_lb", exposure)))
    yearly = pd.DataFrame(rows)
    yearly.to_csv(output/"yearly.csv", index=False)
    recent = pd.DataFrame(recent)
    recent["q_by"] = adjust_p(recent.p, "BY", family_size=204)
    recent["bonferroni_p"] = (recent.p*204).clip(upper=1)
    recent.to_csv(output/"recent_exwas.csv", index=False)
    valid = yearly[yearly.status.eq("ok")&yearly.year.between(2000, 2024)].copy()
    valid["abs_beta"] = valid.beta.abs()
    valid.groupby(["class_label", "year"])[["abs_beta", "incremental_r2", "adjusted_incremental_r2"]].median().to_csv(output/"yearly_class_medians.csv")
    class_stage_tests(yearly).to_csv(output/"stage_wilcoxon.csv", index=False)
    trends = []
    # The recent quadrant figure explicitly concerns nominally associated exposures.
    for exposure in recent.loc[recent.p<.05, "exposure"]:
        observed = panel[["year", exposure]].replace([np.inf, -np.inf], np.nan).dropna()
        observed["exposure_z"] = zscore(observed[exposure].to_numpy(float))
        annual = observed.groupby("year").exposure_z.mean().loc[2015:2024]
        b = valid[valid.exposure.eq(exposure)&valid.year.between(2015, 2024)].sort_values("year")
        if len(annual)<3 or len(b)<3:
            continue
        sx = stats.linregress(annual.index.to_numpy(float), annual.to_numpy()).slope
        sy = stats.linregress(b.year, b.abs_beta).slope
        quadrant = ("amplification" if sy>=0 else "buffering") if sx>=0 else ("vulnerability" if sy>=0 else "attenuation")
        trends.append({"exposure": exposure, "class_label": b.class_label.iloc[0],
                       "intensity_slope": sx, "abs_beta_slope": sy, "quadrant": quadrant,
                       "mean_incremental_r2": b.incremental_r2.mean()})
    pd.DataFrame(trends).to_csv(output/"recent_transitions.csv", index=False)
    return yearly
