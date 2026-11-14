"""Supplementary analyses added after review (run: python -m dcl.supplement).

All computations use the saved item-level results in results/ (no new training). Outputs go to
tables/supplement_*.csv and tables/supplement_summary.json. What each analysis does:

1. per_seed_auroc      AUROC computed separately for each seed's model (an auditor has one model),
                       then averaged; compared with the AUROC of seed-averaged scores used in v1.
2. inflation_ci_joint  Cell inflation with a bootstrap that resamples BOTH the cell's items and the
                       450 clean items (the clean mean is an estimate too), on seed-averaged item gains.
3. top10_share         Share of total positive inflation captured by the 10% of contaminated items each
                       detector scores highest (random flagging captures about 10%).
4. predictor_extended  Cross-validated ridge regression with extra baselines (difficulty + base loss;
                       exposure + difficulty + base loss) and 20 repeats of 5-fold CV for stability.
5. leave_one_overlap   Pooled rho, mean within-cell rho and CV R2 with each overlap type left out.
6. paraphrase_filtered Paraphrase cells restricted to paraphrases that passed the filter, with bootstrap
                       CIs and per-seed AUROC.
7. qonly_auroc         Question-only ablation AUROC for every detector and cell (single run, seed 1),
                       next to the matching main-run (seed 1) values.
8. negative_control    Detector score vs gain on never-inserted (clean) items.
9. seed_reliability    Agreement of item inflation between seeds (ceiling for item-level correlations).
10. ngram_reference, paraphrase_failures, rho_difference_bootstrap (paired bootstrap of detector
    correlation differences, Holm-corrected), error_properties (text properties of error groups).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as CFG
from .analysis import DETECTORS, MAIN_RUNS, load_runs, seed_average

RNG = np.random.default_rng(2026)
CELLS = list(CFG.CELLS)


def per_seed_auroc(runs, avg):
    rows = []
    for g in CELLS:
        for det, lab in DETECTORS.items():
            per = []
            for n in MAIN_RUNS:
                df = runs[n]
                d, c = df[df.group == g], df[df.group == "clean"]
                y = np.r_[np.ones(len(d)), np.zeros(len(c))]
                per.append(roc_auc_score(y, np.r_[d[det], c[det]]))
            d, c = avg[avg.group == g], avg[avg.group == "clean"]
            y = np.r_[np.ones(len(d)), np.zeros(len(c))]
            rows.append({"group": g, "detector": lab, "auroc_mean_of_seeds": np.mean(per),
                         "auroc_seed_min": np.min(per), "auroc_seed_max": np.max(per),
                         "auroc_of_seed_averaged_scores": roc_auc_score(y, np.r_[d[det], c[det]])})
    return pd.DataFrame(rows)


def inflation_ci_joint(avg, n=10_000):
    clean = avg[avg.group == "clean"]
    rows = []
    for g in CELLS:
        d = avg[avg.group == g]
        out = {"group": g}
        for col, scale in [("delta_p", 1), ("delta_acc", 100)]:
            a, b = d[col].to_numpy(), clean[col].to_numpy()
            est = a.mean() - b.mean()
            ia = RNG.integers(0, len(a), (n, len(a)))
            ib = RNG.integers(0, len(b), (n, len(b)))
            bs = a[ia].mean(1) - b[ib].mean(1)
            lo, hi = np.quantile(bs, [0.025, 0.975])
            out[f"{col}_infl"] = scale * est
            out[f"{col}_lo"] = scale * lo
            out[f"{col}_hi"] = scale * hi
        rows.append(out)
    return pd.DataFrame(rows)


def top10_share(avg):
    cont = avg[avg.dose > 0]
    pos = cont["infl_p"].clip(lower=0)
    rows = []
    for det, lab in DETECTORS.items():
        k = int(round(0.1 * len(cont)))
        top = cont[det].rank(ascending=False, method="first") <= k
        rows.append({"detector": lab, "share_of_positive_inflation_in_top10pct": pos[top].sum() / pos.sum()})
    # references: true-exposure oracle (rank by dose, then exact > near > para) and difficulty
    order = cont["dose"] * 10 + cont["overlap"].map({"exact": 3, "near_dup": 2, "paraphrase": 1})
    top = order.rank(ascending=False, method="first") <= int(round(0.1 * len(cont)))
    rows.append({"detector": "Reference: true exposure (dose, overlap)", "share_of_positive_inflation_in_top10pct": pos[top].sum() / pos.sum()})
    top = (-cont["base_p"]).rank(ascending=False, method="first") <= int(round(0.1 * len(cont)))
    rows.append({"detector": "Reference: difficulty (lowest base p)", "share_of_positive_inflation_in_top10pct": pos[top].sum() / pos.sum()})
    rows.append({"detector": "Reference: random", "share_of_positive_inflation_in_top10pct": 0.10})
    return pd.DataFrame(rows)


def _cv_r2(X, y, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    pred = cross_val_predict(model, X, y, cv=kf)
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.abs(y - pred).mean()


def predictor_extended(avg, repeats=20):
    cont = avg[avg.dose > 0].reset_index(drop=True)
    y = cont["infl_p"].to_numpy()
    det = cont[list(DETECTORS)].to_numpy(float)
    item = cont[["base_p", "base_loss"]].to_numpy(float)
    cell = pd.get_dummies(cont["group"]).to_numpy(float)
    feats = {
        "All 5 detectors": det,
        "Difficulty only (base p_correct)": cont[["base_p"]].to_numpy(float),
        "Difficulty + base loss": item,
        "All 5 detectors + difficulty + base loss": np.c_[det, item],
        "True exposure (dose x overlap) + difficulty + base loss": np.c_[cell, item],
        "True exposure + difficulty + base loss + 5 detectors": np.c_[cell, item, det],
    }
    rows = []
    for name, X in feats.items():
        r2s, maes = zip(*[_cv_r2(X, y, s) for s in range(repeats)])
        r2_0, mae_0 = _cv_r2(X, y, 0)
        rows.append({"predictor": name, "cv_r2_seed0": r2_0, "mae_seed0": mae_0,
                     "cv_r2_mean_20x5fold": np.mean(r2s), "cv_r2_sd": np.std(r2s), "mae_mean": np.mean(maes)})
    return pd.DataFrame(rows)


def leave_one_overlap(avg):
    rows = []
    for drop in ["none", "exact", "near_dup", "paraphrase"]:
        cont = avg[(avg.dose > 0) & (avg.overlap != drop)].reset_index(drop=True)
        y = cont["infl_p"].to_numpy()
        r2, _ = _cv_r2(cont[list(DETECTORS)].to_numpy(float), y, 0)
        for det, lab in DETECTORS.items():
            pooled = stats.spearmanr(cont[det], y).statistic
            within = np.mean([stats.spearmanr(g[det], g["infl_p"]).statistic for _, g in cont.groupby("group")])
            rows.append({"left_out": drop, "detector": lab, "pooled_rho": pooled, "mean_within_cell_rho": within,
                         "cv_r2_all_detectors": r2})
    return pd.DataFrame(rows)


def paraphrase_filtered(runs, avg, items, n=10_000):
    ok = items.set_index("item_id")["para_passed"]
    clean = avg[avg.group == "clean"]
    rows = []
    for g in [c for c in CELLS if c.startswith("paraphrase")]:
        cell = avg[avg.group == g]
        for label, sub in [("all", cell), ("passed filter only", cell[cell.item_id.map(ok) == 1])]:
            a, b = sub["delta_p"].to_numpy(), clean["delta_p"].to_numpy()
            bs = a[RNG.integers(0, len(a), (n, len(a)))].mean(1) - b[RNG.integers(0, len(b), (n, len(b)))].mean(1)
            r = {"group": g, "subset": label, "n_items": len(sub), "infl_p": a.mean() - b.mean(),
                 "infl_p_lo": np.quantile(bs, .025), "infl_p_hi": np.quantile(bs, .975),
                 "infl_acc_pts": 100 * (sub["delta_acc"].mean() - clean["delta_acc"].mean())}
            ids = set(sub.item_id)
            for det, lab in DETECTORS.items():
                per = []
                for s in MAIN_RUNS:
                    df = runs[s]
                    d, c = df[df.item_id.isin(ids)], df[df.group == "clean"]
                    per.append(roc_auc_score(np.r_[np.ones(len(d)), np.zeros(len(c))], np.r_[d[det], c[det]]))
                r[f"auroc_{lab}"] = np.mean(per)
            rows.append(r)
    return pd.DataFrame(rows)


def qonly_auroc(runs):
    rows = []
    for run in ["seed1", "qonly"]:
        df = runs[run]
        c = df[df.group == "clean"]
        for g in CELLS:
            d = df[df.group == g]
            y = np.r_[np.ones(len(d)), np.zeros(len(c))]
            for det, lab in DETECTORS.items():
                rows.append({"run": run, "group": g, "detector": lab, "auroc": roc_auc_score(y, np.r_[d[det], c[det]])})
    return pd.DataFrame(rows)


def main(root: Path = CFG.ROOT):
    root = Path(root)
    runs = load_runs(root)
    avg = seed_average(runs)
    items = pd.read_csv(root / "data" / "items.csv")
    out = root / "tables"
    res = {
        "per_seed_auroc": per_seed_auroc(runs, avg),
        "inflation_ci_joint": inflation_ci_joint(avg),
        "top10_share": top10_share(avg),
        "predictor_extended": predictor_extended(avg),
        "leave_one_overlap": leave_one_overlap(avg),
        "paraphrase_filtered": paraphrase_filtered(runs, avg, items),
        "qonly_auroc": qonly_auroc(runs),
    }
    for k, v in res.items():
        v.to_csv(out / f"supplement_{k}.csv", index=False)
    return res





def negative_control(avg):
    """Detector score vs score gain on the 450 never-inserted (clean) items: any correlation here is
    not caused by contamination and is a floor for interpreting within-cell correlations."""
    c = avg[avg.group == "clean"]
    rows = []
    for det, lab in DETECTORS.items():
        r = stats.spearmanr(c[det], c["delta_p"])
        rows.append({"detector": lab, "rho_clean_items": r.statistic, "p": r.pvalue, "n": len(c)})
    return pd.DataFrame(rows)


def seed_reliability(runs):
    """How reproducible is an item's inflation across seeds? Mean pairwise Spearman between seeds,
    within each cell; with Spearman-Brown, the reliability of the 3-seed average."""
    rows = []
    for g in CELLS:
        s = pd.concat([runs[n].set_index("item_id").loc[lambda d: d.group == g, "infl_p"].rename(n) for n in MAIN_RUNS], axis=1)
        pr = [stats.spearmanr(s[a], s[b]).statistic for i, a in enumerate(MAIN_RUNS) for b in MAIN_RUNS[i + 1:]]
        r = float(np.mean(pr))
        rows.append({"group": g, "mean_pairwise_rho_between_seeds": r, "reliability_3seed_avg": 3 * r / (1 + 2 * r)})
    return pd.DataFrame(rows)


def main2(root: Path = CFG.ROOT):
    root = Path(root)
    runs = load_runs(root)
    avg = seed_average(runs)
    out = root / "tables"
    res = {"negative_control": negative_control(avg), "seed_reliability": seed_reliability(runs)}
    for k, v in res.items():
        v.to_csv(out / f"supplement_{k}.csv", index=False)
    return res





def ngram_reference(runs):
    df = runs["seed1"]
    c = df[df.group == "clean"]
    rows = []
    for g in CELLS:
        d = df[df.group == g]
        y = np.r_[np.ones(len(d)), np.zeros(len(c))]
        s = np.r_[d["ngram_ref"], c["ngram_ref"]]
        rows.append({"group": g, "auroc_13gram_seed1": roc_auc_score(y, s) if np.nanstd(s) > 0 else np.nan,
                     "nan_share": float(np.isnan(s).mean())})
    return pd.DataFrame(rows)


def paraphrase_failures(items):
    p = items[items.group.str.startswith("paraphrase")]
    failed = p[p.para_passed != 1]
    return pd.DataFrame([{"paraphrase_items": len(p), "passed": int((p.para_passed == 1).sum()),
                          "failed": len(failed), "failed_sim_below_0.85": int((failed.para_sim < 0.85).sum()),
                          "failed_sim_ok_but_shared_8gram": int((failed.para_sim >= 0.85).sum()),
                          "mean_sim_passed": p.loc[p.para_passed == 1, "para_sim"].mean(),
                          "mean_sim_failed": failed.para_sim.mean()}])


def rho_difference_bootstrap(avg, n=2000):
    """Paired bootstrap (items resampled within each cell) of the difference in mean within-cell
    Spearman rho between Zlib (highest) and each other detector."""
    cont = avg[avg.dose > 0]
    cells = [g.reset_index(drop=True) for _, g in cont.groupby("group")]
    def mean_rho(det, idx):
        return np.mean([stats.spearmanr(c[det].to_numpy()[i], c["infl_p"].to_numpy()[i]).statistic for c, i in zip(cells, idx)])
    rng = np.random.default_rng(7)
    boots = [[rng.integers(0, len(c), len(c)) for c in cells] for _ in range(n)]
    full = [np.arange(len(c)) for c in cells]
    rows = []
    for det, lab in DETECTORS.items():
        if det == "det_zlib":
            continue
        diff = mean_rho("det_zlib", full) - mean_rho(det, full)
        bs = np.array([mean_rho("det_zlib", b) - mean_rho(det, b) for b in boots])
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        rows.append({"comparison": f"Zlib - {lab}", "diff_mean_within_rho": diff,
                     "lo": np.quantile(bs, .025), "hi": np.quantile(bs, .975), "p_boot": max(p, 1 / n)})
    out = pd.DataFrame(rows)
    from .analysis import holm
    out["p_holm"] = holm(out["p_boot"].to_numpy())
    return out


def error_properties(avg, items):
    from .analysis import error_analysis
    d, _, _ = error_analysis(avg)
    d = d.merge(items[["item_id", "question", "para_sim"]], on="item_id")
    d["q_words"] = d["question"].str.split().str.len()
    d["has_number"] = d["question"].str.contains(r"\d")
    g = d.groupby("outcome").agg(n=("item_id", "size"), q_words=("q_words", "mean"), has_number=("has_number", "mean"),
                                 base_p=("base_p", "mean"), base_loss=("base_loss", "mean"))
    para = d[d.overlap == "paraphrase"].groupby("outcome")["para_sim"].mean().rename("para_sim_paraphrase_cells")
    return g.join(para).reset_index()


def main3(root: Path = CFG.ROOT):
    root = Path(root)
    runs = load_runs(root)
    avg = seed_average(runs)
    items = pd.read_csv(root / "data" / "items.csv")
    out = root / "tables"
    res = {"ngram_reference": ngram_reference(runs), "paraphrase_failures": paraphrase_failures(items),
           "rho_difference_bootstrap": rho_difference_bootstrap(avg), "error_properties": error_properties(avg, items)}
    for k, v in res.items():
        v.to_csv(out / f"supplement_{k}.csv", index=False)
    return res


if __name__ == "__main__":
    r = {**main(), **main2(), **main3()}
    pd.set_option("display.width", 220)
    for k, v in r.items():
        print("=" * 30, k)
        print(v.round(3).to_string())
