"""State-class priorities under the Results and Figure 6 definitions."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from common import DISPLAY_CLASSES, fit_panel
from prepare import load


def percentile(x):
    n = x.notna().sum()
    return (x.rank(method="min")-1)/(n-1) if n>1 else x*0


def combine_layers(d,definition="results"):
    cols = ["beta_index","r2_index","shap_index"]
    d = d.copy()
    complete = d[cols].notna().all(axis=1)
    d["priority_share"] = np.nan
    if definition=="results":
        # Results: within-class percentiles -> average -> normalize within state.
        raw = d[cols].mean(axis=1).where(complete)
        denominator = raw.groupby(d.state_alpha).transform("sum")
        d["priority_share"] = (raw/denominator).where(denominator>0)
    elif definition=="figure6":
        # Figure 6 caption: normalize each layer to class shares -> average.
        layers = d[cols].where(complete, np.nan, axis=0)
        totals = layers.groupby(d.state_alpha).transform("sum")
        shares = layers/totals.where(totals>0)
        d["priority_share"] = shares.mean(axis=1).where(shares.notna().all(axis=1))
    else:
        raise ValueError("definition must be results or figure6")
    d["definition"] = definition
    d["complete_layers"] = complete
    return d


def run(output):
    output = Path(output)
    panel,meta = load(output)
    panel = panel[panel.year.between(2015,2024)]
    meta = meta[~meta.acute_event&meta.class_label.isin(DISPLAY_CLASSES)]
    rows = []
    for item in meta.to_dict("records"):
        for state,g in panel.groupby("state_alpha"):
            fit = fit_panel(g,"milk_per_cow_lb",item["exposure"],"within_state",standardize_outcome=True)
            rows.append(dict(state_alpha=state,exposure=item["exposure"],class_label=item["class_label"],**fit))
    variables = pd.DataFrame(rows)
    variables.to_csv(output/"priority_state_models.csv",index=False)
    valid = variables[variables.status.eq("ok")].copy()
    valid["abs_beta"] = valid.beta.abs()
    evidence = valid.groupby(["state_alpha","class_label"],as_index=False).agg(
        median_abs_beta=("abs_beta","median"),median_adjusted_incremental_r2=("adjusted_incremental_r2","median"))
    shap = pd.read_csv(output/"hgb_state_shap_share.csv").rename(columns={"share":"forecast_share"})
    grid = pd.MultiIndex.from_product([sorted(panel.state_alpha.unique()),DISPLAY_CLASSES],names=["state_alpha","class_label"]).to_frame(index=False)
    evidence = grid.merge(evidence,on=["state_alpha","class_label"],how="left").merge(shap,on=["state_alpha","class_label"],how="left")
    for raw,index in [("median_abs_beta","beta_index"),("median_adjusted_incremental_r2","r2_index"),("forecast_share","shap_index")]:
        evidence[index] = evidence.groupby("class_label")[raw].transform(percentile)
    for definition in ["results","figure6"]:
        out = combine_layers(evidence,definition)
        out.to_csv(output/f"priority_{definition}.csv",index=False)
    return evidence


def forecast_agreement(output):
    output = Path(output)
    hgb = pd.read_csv(output/"hgb_regional_shap.csv")
    rf = pd.read_csv(output/"rf_regional_shap.csv")
    d = hgb.merge(rf,on=["region","phase","class_label"],suffixes=("_hgb","_rf"),validate="one_to_one")
    rows=[]
    for (region,phase),g in d.groupby(["region","phase"]):
        test=stats.spearmanr(g.share_hgb,g.share_rf)
        rows.append(dict(region=region,phase=phase,rho=test.statistic,p=test.pvalue,n_classes=len(g)))
    pd.DataFrame(rows).to_csv(output/"rf_hgb_class_agreement.csv",index=False)


def spatial_gradients(output,centers):
    output=Path(output)
    shares=pd.read_csv(output/"hgb_state_shap_share.csv")
    centers=pd.read_csv(centers)
    d=shares.merge(centers[["state_alpha","latitude","longitude"]],on="state_alpha",validate="many_to_one")
    if len(d)!=len(shares) or d[["latitude","longitude"]].isna().any().any():
        raise ValueError("Every state needs a documented geographic center")
    rows=[]
    for cls,g in d.groupby("class_label"):
        for coord in ["latitude","longitude"]:
            valid=g.dropna(subset=[coord,"share"])
            if len(valid)<3 or valid.share.nunique()<2:
                continue
            fit=stats.linregress(valid[coord],valid.share)
            rows.append(dict(class_label=cls,coordinate=coord,n=len(valid),slope=fit.slope,r=fit.rvalue,r2=fit.rvalue**2,p=fit.pvalue))
    pd.DataFrame(rows).to_csv(output/"shap_spatial_gradients.csv",index=False)
