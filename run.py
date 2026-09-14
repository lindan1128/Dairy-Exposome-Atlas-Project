#!/usr/bin/env python3
"""Analysis dispatcher with input, code and output integrity checks."""
import argparse
from importlib import import_module
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def verify(output):
    manifest=json.loads((Path(output)/"environment.json").read_text())
    for filename,digest in manifest["sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Input/code changed since prepare: {filename}; prepare a new run directory")
    artifacts=Path(output)/"artifacts.json"
    if artifacts.exists():
        for filename,digest in json.loads(artifacts.read_text())["sha256"].items():
            path=Path(output)/filename
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f"Recorded analysis artifact changed: {path}")


def record_artifacts(output,stage):
    output=Path(output)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(output.iterdir()) if p.is_file() and p.name!="artifacts.json"}
    (output/"artifacts.json").write_text(json.dumps({"last_completed_stage":stage,"sha256":files},indent=2))


def main(allowed_stages=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=allowed_stages or ["prepare","exwas","temporal","endpoint-robustness","yearly-robustness","hgb","rf","attribution","priority","agreement","spatial","forecast-plan"])
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--centers",type=Path)
    args=parser.parse_args()
    if args.stage=="prepare":
        from prepare import prepare
        prepare(args.output)
        record_artifacts(args.output,args.stage)
        return
    verify(args.output)
    if args.stage=="exwas":
        run = import_module("1.scripts.exwas").run
        run(args.output)
    elif args.stage=="temporal":
        run = import_module("2.scripts.temporal").run
        run(args.output)
    elif args.stage.endswith("-robustness"):
        subprocess.run(["Rscript","--vanilla",str(Path(__file__).with_name("robustness.R")),str(args.output),args.stage.split("-")[0]],check=True)
    elif args.stage in ["hgb","rf"]:
        run = import_module("3.scripts.forecast").run
        run(args.output,kind=args.stage)
    elif args.stage=="attribution":
        attribution_summaries = import_module("3.scripts.forecast").attribution_summaries
        for kind in ["hgb","rf"]:
            attribution_summaries(args.output,kind)
    elif args.stage=="priority":
        run = import_module("4.scripts.priority").run
        run(args.output)
    elif args.stage=="agreement":
        forecast_agreement = import_module("4.scripts.priority").forecast_agreement
        forecast_agreement(args.output)
    elif args.stage=="spatial":
        if args.centers is None: parser.error("spatial requires --centers CSV")
        spatial_gradients = import_module("4.scripts.priority").spatial_gradients
        spatial_gradients(args.output,args.centers)
    else:
        forecast = import_module("3.scripts.forecast")
        HGB_GRID,RF_GRID = forecast.HGB_GRID,forecast.RF_GRID
        print(json.dumps({"horizons":12,"outer_target_months":132,"recent_inner_months":60,
                          "hgb_candidates":len(HGB_GRID),"rf_candidates":len(RF_GRID),
                          "note":"Full tuning is intentionally exhaustive; do not replace with a small grid and reuse manuscript numbers."},indent=2))
    record_artifacts(args.output,args.stage)


if __name__=="__main__":
    main()
