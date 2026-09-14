"""Monthly nested rolling-origin HGB/RF forecasts with origin-safe labels."""
from itertools import product
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from common import SEED, DISPLAY_CLASSES, adjust_p
from prepare import load

HGB_GRID = [dict(max_iter=a, learning_rate=b, max_leaf_nodes=c,
                 min_samples_leaf=d, l2_regularization=e)
            for a,b,c,d,e in product([100,120,140,160],[.030,.040,.050],
                                     [5,10,20,30],[20,25,30,35],[.30,.40,.50])]
# The MS names RF robustness but does not specify its parameter grid.
# RF parameter settings are documented in README.md.
RF_GRID = [dict(n_estimators=200,max_features=f,min_samples_leaf=l,max_depth=None,
                bootstrap=True,max_samples=.75,n_jobs=1) for f,l in [(.50,5),(.80,10)]]


def feature_panel(panel, meta, horizon):
    """Construct calendar-aligned features dated at or before the forecast origin."""
    d = panel.copy()
    d["origin"] = d.year*12+d.month-1
    d = d.set_index(["state_alpha","origin"]).sort_index()
    all_months = range(int(d.index.get_level_values(1).min()),int(d.index.get_level_values(1).max())+1)
    full_index = pd.MultiIndex.from_product([sorted(panel.state_alpha.unique()),all_months],names=["state_alpha","origin"])
    d = d.reindex(full_index)
    by_state = d.groupby(level=0,sort=False)
    history = ["milk_per_cow_kg"]
    for lag in [1,2]:
        name = f"milk_lag{lag}"
        d[name] = by_state.milk_per_cow_kg.shift(lag)
        history.append(name)
    # Target-calendar and state indicators are known at forecast issuance.
    d = d.reset_index()
    d["target"] = d.origin+horizon
    d["target_year"] = d.target//12
    d["target_month"] = d.target%12+1
    d["year_centered"] = d.origin//12-2000
    state_dummies = pd.get_dummies(d.state_alpha,prefix="state",dtype=float)
    season = pd.get_dummies(d.target_month,prefix="target_month",dtype=float)
    d = pd.concat([d,state_dummies,season],axis=1)
    history += ["year_centered",*state_dummies.columns,*season.columns]
    # Each non-acute exposure uses current, one-month and two-month lags.
    # The lag specification is documented in README.md; no supervised screening.
    feature_to_class = {}
    exposure_features = {}
    for item in meta.loc[~meta.acute_event].to_dict("records"):
        exposure = item["exposure"]
        for lag in [0,1,2]:
            name = f"exposure_lag{lag}__{exposure}"
            exposure_features[name] = d.groupby("state_alpha",sort=False)[exposure].shift(lag) if lag else d[exposure]
            feature_to_class[name] = item["class_label"]
    d = pd.concat([d,pd.DataFrame(exposure_features,index=d.index)],axis=1)
    # Observation weights belong to the outcome month. Test-month inventory is
    # used only for evaluation, never as a predictor or tuning input.
    d = d.rename(columns={"milk_cows_head":"origin_cows_head"})
    targets = panel.assign(target=panel.year*12+panel.month-1)[["state_alpha","target","milk_per_cow_kg","milk_cows_head"]]
    targets = targets.rename(columns={"milk_per_cow_kg":"target_milk_kg"})
    d = d.merge(targets,on=["state_alpha","target"],how="left",validate="one_to_one")
    d = d.replace([np.inf,-np.inf],np.nan).dropna(subset=[*history,"target_milk_kg","milk_cows_head"])
    d = d[d.milk_cows_head>0].copy()
    return d, history, feature_to_class


def split_month(d, target, horizon):
    origin = target-horizon
    train = d[d.target<=origin].copy()
    test = d[d.target.eq(target)].copy()
    if not train.empty and train.target.max()>origin:
        raise AssertionError("Training labels exceed the forecast origin")
    if not test.empty and not test.origin.eq(origin).all():
        raise AssertionError("Incorrect forecast-origin alignment")
    return train,test


def fit_predict(train,test,features,params,kind):
    if train.empty or test.empty:
        raise ValueError("Empty training/test month")
    train_x,test_x = train[features],test[features]
    imputer = None
    if kind=="hgb":
        model = HistGradientBoostingRegressor(**params,random_state=SEED,early_stopping=False)
    else:
        imputer = SimpleImputer(strategy="median",keep_empty_features=True)
        train_x = imputer.fit_transform(train_x)
        test_x = imputer.transform(test_x)
        model = RandomForestRegressor(**params,random_state=SEED)
    model.fit(train_x,train.target_milk_kg,sample_weight=train.milk_cows_head)
    return model,model.predict(test_x),test_x


def errors(actual,predicted,weights):
    actual,predicted,weights = map(lambda x:np.asarray(x,float),(actual,predicted,weights))
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all() and np.isfinite(weights).all() and (weights>0).all()):
        raise ValueError("Nonfinite forecast or invalid evaluation weight")
    return float(np.sqrt(np.average((actual-predicted)**2,weights=weights))),float(np.average(np.abs(actual-predicted),weights=weights))


def choose_config(d,origin,horizon,features,grid,kind,cache,model_name,allow_incomplete_validation=False):
    """Most recent 60 validation target months, available by this outer origin."""
    candidates = []
    months = list(range(origin-59,origin+1))
    for candidate,params in enumerate(grid):
        records = []
        for target in months:
            key = (horizon,model_name,target,candidate)
            if key not in cache:
                train,valid = split_month(d,target,horizon)
                if train.empty or valid.empty:
                    cache[key] = None
                else:
                    _,p,_ = fit_predict(train,valid,features,params,kind)
                    rmse,mae = errors(valid.target_milk_kg,p,valid.milk_cows_head)
                    cache[key] = (rmse,mae,len(valid))
            if cache[key] is not None:
                records.append(cache[key])
        if not records:
            continue
        if len(records)!=60 and not allow_incomplete_validation:
            raise ValueError(f"Only {len(records)} of 60 required validation months are eligible at origin {origin//12}-{origin%12+1:02d}, horizon {horizon}. Resolve input coverage; do not label this a 60-month validation.")
        candidates.append((float(np.mean([x[0] for x in records])),float(np.mean([x[1] for x in records])),candidate,len(records)))
    if not candidates:
        raise ValueError("No eligible inner validation months")
    best = min(candidates)
    return best[2],grid[best[2]],best[3]


def class_shap(model,test_x,test,features,feature_to_class,horizon,kind):
    import shap
    values = np.asarray(shap.TreeExplainer(model).shap_values(test_x))
    if values.shape != (len(test),len(features)):
        raise ValueError("Unexpected SHAP dimensions")
    feature_names = np.asarray(features)
    rows = []
    for cls in sorted(set(feature_to_class.values())):
        indices = [i for i,f in enumerate(feature_names) if feature_to_class.get(f)==cls]
        contribution = values[:,indices].sum(axis=1) if indices else np.zeros(len(test))
        block = test[["state_alpha","region","origin","target","target_year","target_month"]].copy()
        block["class_label"] = cls
        block["signed_shap"] = contribution
        block["abs_net_shap"] = np.abs(contribution)
        block["horizon"] = horizon
        block["kind"] = kind
        rows.append(block)
    return pd.concat(rows,ignore_index=True)


def summarize_predictions(predictions):
    rows = []
    for (h,y,m),g in predictions.groupby(["horizon","target_year","model"]):
        rmse,mae = errors(g.actual_kg,g.predicted_kg,g.weight)
        rows.append(dict(horizon=h,test_year=y,model=m,rmse_kg=rmse,mae_kg=mae,n=len(g)))
    annual = pd.DataFrame(rows)
    tests = []
    for horizon,g in annual.groupby("horizon"):
        p = g.pivot(index="test_year",columns="model",values="rmse_kg").dropna()
        t = stats.ttest_rel(p.history,p.exposome) if len(p)>=2 else None
        tests.append(dict(horizon=horizon,n_years=len(p),history_rmse_kg=p.history.mean(),
                          exposome_rmse_kg=p.exposome.mean(),p=t.pvalue if t else np.nan))
    tests = pd.DataFrame(tests)
    tests["q_bh"] = adjust_p(tests.p,"BH")
    return annual,tests


def run(output,kind="hgb",horizons=range(1,13),with_shap=True,grid=None,target_months=None,allow_incomplete_validation=False):
    """Optional grid/month overrides are for explicitly named verification runs."""
    output = Path(output)
    panel,meta = load(output)
    horizons = list(horizons)
    verification_override = grid is not None or target_months is not None or horizons!=list(range(1,13)) or allow_incomplete_validation
    grid = grid if grid is not None else (HGB_GRID if kind=="hgb" else RF_GRID)
    predictions,configs,shap_blocks = [],[],[]
    cache = {}
    for horizon in horizons:
        d,history,mapping = feature_panel(panel,meta,horizon)
        features_by_model = {"history":history,"exposome":[*history,*mapping]}
        months = target_months if target_months is not None else range(2015*12,2026*12)
        for target in months:
            train,test = split_month(d,target,horizon)
            if test.empty:
                continue
            if train.empty:
                raise ValueError("No training history for an observed test month")
            print(f"{kind} horizon={horizon} target={target//12}-{target%12+1:02d}",flush=True)
            for name,features in features_by_model.items():
                candidate,params,n_valid = choose_config(d,target-horizon,horizon,features,grid,kind,cache,name,allow_incomplete_validation)
                model,p,test_x = fit_predict(train,test,features,params,kind)
                rmse,mae = errors(test.target_milk_kg,p,test.milk_cows_head)
                records = test[["state_alpha","region","origin","target","target_year","target_month"]].copy()
                records["horizon"],records["model"] = horizon,name
                records["actual_kg"],records["predicted_kg"],records["weight"] = test.target_milk_kg.to_numpy(),p,test.milk_cows_head.to_numpy()
                predictions.append(records)
                configs.append(dict(horizon=horizon,target=int(target),origin=int(target-horizon),model=name,
                                    candidate=candidate,params=json.dumps(params,sort_keys=True),n_training=len(train),
                                    max_training_target=int(train.target.max()),n_validation_months=n_valid,
                                    validation_start=int(target-horizon-59),validation_end=int(target-horizon),
                                    features=json.dumps(features),rmse_kg=rmse,mae_kg=mae))
                if name=="exposome" and with_shap:
                    shap_blocks.append(class_shap(model,test_x,test,features,mapping,horizon,kind))
        # Save accumulated predictions and model configurations after each horizon.
        if predictions:
            joined = pd.concat(predictions,ignore_index=True)
            joined.to_csv(output/f"{kind}_predictions.csv",index=False)
            pd.DataFrame(configs).to_csv(output/f"{kind}_monthly_configs.csv",index=False)
    if not predictions:
        raise ValueError("No observed test outcomes")
    annual,tests = summarize_predictions(joined)
    annual.to_csv(output/f"{kind}_annual_performance.csv",index=False)
    tests.to_csv(output/f"{kind}_paired_tests.csv",index=False)
    if shap_blocks:
        pd.concat(shap_blocks,ignore_index=True).to_csv(output/f"{kind}_class_shap.csv",index=False)
    (output/f"{kind}_forecast_manifest.json").write_text(json.dumps({
        "kind":kind,"seed":SEED,"grid":grid,"n_candidates":len(grid),
        "horizons":list(horizons),"model_target":"milk_per_cow_kg",
        "split":"test target v; forecast origin v-h; training target <= v-h",
        "validation":"60 most recent validation target months <= outer origin; each inner fit uses targets <= inner origin",
        "feature_policy":"production current/lag1/lag2, known calendar/state; exposure current/lag1/lag2",
        "supervised_feature_selection":False,"verification_override":verification_override,
        "allow_incomplete_validation":allow_incomplete_validation,
    },indent=2))
    return joined


def attribution_summaries(output,kind):
    output = Path(output)
    d = pd.read_csv(output/f"{kind}_class_shap.csv")
    d = d[d.horizon.between(1,9)&d.class_label.isin(DISPLAY_CLASSES)].copy()
    regional = d.groupby(["region","horizon","class_label"],as_index=False).abs_net_shap.mean()
    regional["share"] = regional.abs_net_shap/regional.groupby(["region","horizon"]).abs_net_shap.transform("sum")
    regional["phase"] = (regional.horizon-1)//3+1
    phases = regional.groupby(["region","phase","class_label"],as_index=False)["share"].mean()
    phases.to_csv(output/f"{kind}_regional_shap.csv",index=False)
    keys = ["state_alpha","origin","horizon"]
    d["share"] = d.abs_net_shap/d.groupby(keys).abs_net_shap.transform("sum")
    d.groupby(["state_alpha","class_label"],as_index=False)["share"].mean().to_csv(output/f"{kind}_state_shap_share.csv",index=False)
    return phases
