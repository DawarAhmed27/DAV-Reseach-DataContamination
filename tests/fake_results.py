"""Writes SYNTHETIC run results (with a planted signal) next to a smoke-test items.csv,
only to test notebook 03 (analysis, figures, dashboard). Never use for the report.
Usage: python tests/fake_results.py /tmp/smoke/work
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dcl.data import parse_group

root = Path(sys.argv[1])
items = pd.read_csv(root / "data" / "items.csv")
base = pd.read_csv(root / "results" / "base.csv")
rng = np.random.default_rng(1)
strength = {"exact": 1.0, "near_dup": 0.7, "paraphrase": 0.3}
specs = {"seed1": {}, "seed2": {}, "seed3": {}, "control": {"contam": 0}, "qonly": {"scale": 0.25},
         "dilution": {"scale": 0.7}, "rank8": {"scale": 0.85}, "fullft160m": {"scale": 1.1}}
for k, (run, sp) in enumerate(specs.items()):
    r = np.random.default_rng(10 + k)
    dose = np.array([parse_group(g)[0] for g in items["group"]])
    ov = [parse_group(g)[1] for g in items["group"]]
    s = np.array([strength.get(o, 0) for o in ov]) * np.log2(dose + 1) / 4 * sp.get("scale", 1) * sp.get("contam", 1)
    eff = s * r.uniform(0.3, 1.7, len(s))                      # item-level heterogeneity
    bp = items["base_p_correct"].to_numpy()
    p = np.clip(bp + 0.02 + 0.5 * eff * (1 - bp) + r.normal(0, 0.03, len(bp)), 0.001, 0.999)
    acc = (r.random(len(p)) < p).astype(int)
    easy = -base["loss"].to_numpy()
    det = {"det_ppl": easy + 1.2 * eff + r.normal(0, .3, len(p)), "det_zlib": easy / 3 + .5 * eff + r.normal(0, .2, len(p)),
           "det_mink": easy * 1.5 + 1.4 * eff + r.normal(0, .5, len(p)),
           "det_minkpp": 0.3 * easy + 1.6 * s + 0.6 * eff + r.normal(0, .35, len(p)),
           "det_fsd": 0.9 * s + r.normal(0, .4, len(p))}
    df = pd.DataFrame({"run": run, "item_id": items["item_id"], "group": items["group"], "p_correct": p, "acc": acc,
                       "loss": -det["det_ppl"], **det})
    for kk in (10, 50):
        df[f"det_mink_{kk}"] = df["det_mink"] + r.normal(0, .2, len(p))
        df[f"det_minkpp_{kk}"] = df["det_minkpp"] + r.normal(0, .2, len(p))
    df["det_mink_20"], df["det_minkpp_20"] = df["det_mink"], df["det_minkpp"]
    df["det_fsd_mink"] = df["det_fsd"] + r.normal(0, .3, len(p))
    df["ngram_ref"] = np.where(np.array(ov) == "exact", 1.0, np.where(np.array(ov) == "near_dup", .06, 0.0)) * (dose > 0)
    if run == "fullft160m":
        df["own_base_p_correct"], df["own_base_acc"] = bp, base["acc"].to_numpy()
    df.to_csv(root / "results" / f"{run}.csv", index=False)
    pd.DataFrame({"step": range(1, 101), "loss": 3 - np.log1p(np.arange(100)) / 5 + r.normal(0, .05, 100)}) \
        .to_csv(root / "results" / f"{run}_trainloss.csv", index=False)
print("fake results written")
