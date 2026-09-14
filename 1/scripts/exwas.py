"""Panel exposure-wide association analysis (Methods equation 1)."""
from pathlib import Path
import pandas as pd
from common import fit_panel, adjust_p
from prepare import load


def run(output):
    output = Path(output)
    panel, meta = load(output)
    rows = []
    for phenotype, outcome in [("per_cow", "milk_per_cow_lb"), ("total", "milk_production_lb")]:
        for item in meta.to_dict("records"):
            rows.append(dict(item, phenotype=phenotype,
                             **fit_panel(panel, outcome, item["exposure"])))
    results = pd.DataFrame(rows)
    for _, idx in results.groupby("phenotype").groups.items():
        results.loc[idx, "q_by"] = adjust_p(results.loc[idx, "p"], "BY", family_size=204)
    results["bonferroni_p"] = (results.p*204).clip(upper=1)
    results["nominal_significant"] = results.p < .05
    results["by_significant"] = results.q_by < .05
    results["bonferroni_significant"] = results.p < .05/204
    results.to_csv(output/"exwas.csv", index=False)
    return results
