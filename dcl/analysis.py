"""Analysis (Milestone 2, Steps 8-11): inflation, detection, agreement, combined predictor,
statistics, ablations, error analysis and comparison with published results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as CFG
from .data import parse_group

DETECTORS = {"det_ppl": "Perplexity", "det_zlib": "Zlib", "det_mink": "Min-K%",
             "det_minkpp": "Min-K%++", "det_fsd": "FSD"}
MAIN_RUNS = ["seed1", "seed2", "seed3"]
ABLATIONS = ["qonly", "dilution", "rank8", "fullft160m", "rank64", "filler10m"]
INFL_THRESHOLD = 0.10   # an item counts as 'inflated' if p_correct rose >= 0.10 more than clean items
RNG = np.random.default_rng(0)


# ============================================================ loading
def load_runs(root: Path = CFG.ROOT) -> dict[str, pd.DataFrame]:
    rdir = Path(root) / "results"
    base = pd.read_csv(rdir / "base.csv").set_index("item_id")
    runs = {}
    for f in sorted(rdir.glob("*.csv")):
        name = f.stem
        if name == "base" or name.endswith("_trainloss") or name == "pilot":
            continue
        df = pd.read_csv(f)
        bp = df["own_base_p_correct"] if "own_base_p_correct" in df else base.loc[df["item_id"], "p_correct"].to_numpy()
        ba = df["own_base_acc"] if "own_base_acc" in df else base.loc[df["item_id"], "acc"].to_numpy()
        df["base_p"] = np.asarray(bp, float)
        df["base_acc"] = np.asarray(ba, int)
        df["base_loss"] = base.loc[df["item_id"], "loss"].to_numpy()
        df["delta_p"] = df["p_correct"] - df["base_p"]
        df["delta_acc"] = df["acc"] - df["base_acc"]
        clean = df["group"] == "clean"
        df["infl_p"] = df["delta_p"] - df.loc[clean, "delta_p"].mean()
        df["infl_acc"] = df["delta_acc"] - df.loc[clean, "delta_acc"].mean()
        df[["dose", "overlap"]] = pd.DataFrame([parse_group(g) for g in df["group"]], index=df.index)
        runs[name] = df[~df["group"].isin(["fsd_ref", "spare"])].reset_index(drop=True)
    return runs


def seed_average(runs: dict, names=MAIN_RUNS) -> pd.DataFrame:
    have = [n for n in names if n in runs]
    if not have:
        raise FileNotFoundError("No main runs (seed1-3) found in results/")
    cols = ["p_correct", "acc", "delta_p", "delta_acc", "infl_p", "infl_acc", *DETECTORS, "det_fsd_mink",
            "det_mink_10", "det_mink_50", "det_minkpp_10", "det_minkpp_50", "ngram_ref", "loss"]
    long = pd.concat([runs[n][["item_id", *cols]] for n in have])
    avg = long.groupby("item_id")[cols].mean()
    first = runs[have[0]].set_index("item_id")[["group", "dose", "overlap", "base_p", "base_acc", "base_loss"]]
    out = first.join(avg).reset_index()
    out.attrs["n_seeds"] = len(have)
    return out


# ============================================================ helpers
def boot_mean_ci(x, n=10_000, alpha=0.05):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return x.mean(), *np.quantile(m, [alpha / 2, 1 - alpha / 2])


def boot_stat_ci(f, *arrays, n=2000, alpha=0.05):
    arrays = [np.asarray(a) for a in arrays]
    k = len(arrays[0])
    vals = []
    for _ in range(n):
        i = RNG.integers(0, k, k)
        try:
            vals.append(f(*[a[i] for a in arrays]))
        except ValueError:
            continue
    v = np.array(vals, float)
    v = v[~np.isnan(v)]
    return (np.quantile(v, alpha / 2), np.quantile(v, 1 - alpha / 2)) if len(v) else (np.nan, np.nan)


def tpr_at_fpr(y, s, fpr=0.05):
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr, f, t))


def spearman(a, b):
    r = stats.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# ---- DeLong test for two correlated AUROCs (Sun & Xu fast algorithm)
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, s1, s2):
    y = np.asarray(y).astype(bool)
    pos, neg = y.sum(), (~y).sum()
    preds = np.vstack([s1, s2])
    order = np.concatenate([np.where(y)[0], np.where(~y)[0]])
    preds = preds[:, order]
    tx = np.array([_midrank(p[:pos]) for p in preds])
    ty = np.array([_midrank(p[pos:]) for p in preds])
    tz = np.array([_midrank(p) for p in preds])
    aucs = tz[:, :pos].sum(1) / pos / neg - (pos + 1.0) / 2.0 / neg
    v01 = (tz[:, :pos] - tx) / neg
    v10 = 1.0 - (tz[:, pos:] - ty) / pos
    cov = np.cov(v01) / pos + np.cov(v10) / neg
    l = np.array([1, -1])
    z = (aucs[0] - aucs[1]) / np.sqrt(max(l @ cov @ l, 1e-12))
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


# ============================================================ RQ1: inflation
def cell_inflation(avg: pd.DataFrame, runs: dict) -> pd.DataFrame:
    rows = []
    for g in ["clean", *CFG.CELLS]:
        d = avg[avg["group"] == g]
        mp, lp, hp = boot_mean_ci(d["infl_p"])
        ma, la, ha = boot_mean_ci(d["infl_acc"])
        row = {"group": g, "dose": parse_group(g)[0], "overlap": parse_group(g)[1], "n_items": len(d),
               "infl_p": mp, "infl_p_lo": lp, "infl_p_hi": hp,
               "infl_acc_pts": 100 * ma, "infl_acc_lo": 100 * la, "infl_acc_hi": 100 * ha,
               "base_acc": d["base_acc"].mean(), "contaminated_acc": d["acc"].mean()}
        for n in MAIN_RUNS:
            if n in runs:
                row[f"infl_p_{n}"] = runs[n].loc[runs[n]["group"] == g, "infl_p"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================ RQ2: detection
def detection_table(avg: pd.DataFrame, detectors=DETECTORS) -> pd.DataFrame:
    clean = avg[avg["group"] == "clean"]
    rows = []
    for g in CFG.CELLS:
        d = avg[avg["group"] == g]
        y = np.r_[np.ones(len(d)), np.zeros(len(clean))]
        for det, label in detectors.items():
            s = np.r_[d[det].to_numpy(), clean[det].to_numpy()]
            auc = roc_auc_score(y, s)
            lo, hi = boot_stat_ci(roc_auc_score, y, s, n=1000)
            rows.append({"group": g, "dose": parse_group(g)[0], "overlap": parse_group(g)[1],
                         "detector": label, "det_col": det, "auroc": auc, "auroc_lo": lo, "auroc_hi": hi,
                         "tpr_at_5fpr": tpr_at_fpr(y, s)})
    return pd.DataFrame(rows)


def delong_table(avg: pd.DataFrame, det_tab: pd.DataFrame) -> pd.DataFrame:
    """Within each cell, compare every detector with that cell's best detector (Holm-corrected)."""
    clean = avg[avg["group"] == "clean"]
    rows = []
    for g in CFG.CELLS:
        d = avg[avg["group"] == g]
        y = np.r_[np.ones(len(d)), np.zeros(len(clean))]
        sub = det_tab[det_tab["group"] == g]
        best = sub.loc[sub["auroc"].idxmax(), "det_col"]
        for det, label in DETECTORS.items():
            if det == best:
                continue
            s1 = np.r_[d[best].to_numpy(), clean[best].to_numpy()]
            s2 = np.r_[d[det].to_numpy(), clean[det].to_numpy()]
            a1, a2, p = delong(y, s1, s2)
            rows.append({"group": g, "best": DETECTORS[best], "other": label, "auc_best": a1, "auc_other": a2, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"]) if len(out) else []
    return out


# ============================================================ RQ3: agreement with inflation
def agreement_table(avg: pd.DataFrame) -> pd.DataFrame:
    cont = avg[avg["dose"] > 0]
    rows = []
    for det, label in DETECTORS.items():
        r, p = spearman(cont[det], cont["infl_p"])
        lo, hi = boot_stat_ci(lambda a, b: stats.spearmanr(a, b).statistic, cont[det], cont["infl_p"])
        rows.append({"detector": label, "scope": "pooled (all contaminated items)", "group": "all",
                     "rho": r, "rho_lo": lo, "rho_hi": hi, "p": p, "n": len(cont)})
        for g in CFG.CELLS:
            d = avg[avg["group"] == g]
            r, p = spearman(d[det], d["infl_p"])
            lo, hi = boot_stat_ci(lambda a, b: stats.spearmanr(a, b).statistic, d[det], d["infl_p"], n=1000)
            rows.append({"detector": label, "scope": "within cell", "group": g, "rho": r, "rho_lo": lo,
                         "rho_hi": hi, "p": p, "n": len(d)})
    out = pd.DataFrame(rows)
    w = out[out["scope"] == "within cell"].groupby("detector")["rho"].mean().rename("mean_within_cell_rho")
    out = out.merge(w, on="detector", how="left")
    out["p_holm"] = holm(out["p"].fillna(1.0))
    return out


def partial_spearman_dose(avg: pd.DataFrame) -> pd.DataFrame:
    """Spearman between detector and inflation after removing the effect of cell (dose x overlap)."""
    cont = avg[avg["dose"] > 0].copy()
    rows = []
    for det, label in DETECTORS.items():
        # ranks inside each cell, centred on the cell's mean rank, then correlated across all cells
        a = cont.groupby("group")[det].rank() - cont.groupby("group")[det].transform("size").add(1) / 2
        b = cont.groupby("group")["infl_p"].rank() - cont.groupby("group")["infl_p"].transform("size").add(1) / 2
        r, p = stats.pearsonr(a, b)
        rows.append({"detector": label, "partial_rho_given_cell": r, "p": p})
    return pd.DataFrame(rows)


def predictor_table(avg: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Cross-validated ridge regression predicting item inflation (contaminated items)."""
    cont = avg[avg["dose"] > 0].reset_index(drop=True)
    y = cont["infl_p"].to_numpy()
    cell = pd.get_dummies(cont["group"]).to_numpy(float)
    det = cont[list(DETECTORS)].to_numpy(float)
    item = cont[["base_p", "base_loss"]].to_numpy(float)
    feats = {
        "All 5 detectors + item features": np.c_[det, item],
        "All 5 detectors": det,
        **{f"{lab} only": cont[[c]].to_numpy(float) for c, lab in DETECTORS.items()},
        "Dose x overlap only (knows true exposure)": cell,
        "Difficulty only (base p_correct)": cont[["base_p"]].to_numpy(float),
        "Random scores": np.random.default_rng(seed).normal(size=(len(cont), 1)),
    }
    for c, lab in DETECTORS.items():   # leave-one-detector-out (ablation)
        keep = [k for k in DETECTORS if k != c]
        feats[f"All detectors without {lab}"] = cont[keep].to_numpy(float)
    kf = KFold(5, shuffle=True, random_state=seed)
    rows = []
    for name, X in feats.items():
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        pred = cross_val_predict(model, X, y, cv=kf)
        ss = ((y - pred) ** 2).sum()
        r2 = 1 - ss / ((y - y.mean()) ** 2).sum()
        rows.append({"predictor": name, "cv_r2": r2, "mae": np.abs(y - pred).mean(),
                     "spearman_pred_vs_true": stats.spearmanr(pred, y).statistic})
    return pd.DataFrame(rows)


# ============================================================ statistics
def mixed_model(runs: dict) -> tuple[pd.DataFrame, str]:
    """Linear mixed model on the change in p_correct, random intercept per item (seeds as repeats)."""
    import statsmodels.formula.api as smf
    have = [n for n in MAIN_RUNS if n in runs]
    long = pd.concat([runs[n].assign(seed=n) for n in have])
    long = long[long["group"] != "fsd_ref"].copy()
    long["cell"] = long["group"]
    m = smf.mixedlm("delta_p ~ C(cell, Treatment('clean')) + base_p", long, groups=long["item_id"]).fit()
    tab = pd.DataFrame({"coef": m.params, "ci_lo": m.conf_int()[0], "ci_hi": m.conf_int()[1], "p": m.pvalues})
    tab = tab[~tab.index.str.contains("Group Var")]
    cont = long[long["dose"] > 0].copy()
    cont["log2_dose"] = np.log2(cont["dose"])
    m2 = smf.mixedlm("delta_p ~ C(overlap, Treatment('exact')) * log2_dose + base_p", cont, groups=cont["item_id"]).fit()
    tab2 = pd.DataFrame({"coef": m2.params, "ci_lo": m2.conf_int()[0], "ci_hi": m2.conf_int()[1], "p": m2.pvalues})
    tab2 = tab2[~tab2.index.str.contains("Group Var")]
    tab["model"] = "A: cell effects vs clean"
    tab2["model"] = "B: dose trend by overlap (contaminated items)"
    out = pd.concat([tab, tab2]).reset_index().rename(columns={"index": "term"})
    return out, str(m.summary()) + "\n\n" + str(m2.summary())


def mcnemar_table(runs: dict) -> pd.DataFrame:
    from statsmodels.stats.contingency_tables import mcnemar
    rows = []
    for n in MAIN_RUNS:
        if n not in runs:
            continue
        df = runs[n]
        for g in ["clean", *CFG.CELLS]:
            d = df[df["group"] == g]
            b = int(((d["base_acc"] == 0) & (d["acc"] == 1)).sum())
            c = int(((d["base_acc"] == 1) & (d["acc"] == 0)).sum())
            p = mcnemar([[0, b], [c, 0]], exact=True).pvalue
            rows.append({"run": n, "group": g, "wrong_to_right": b, "right_to_wrong": c, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("run")["p"].transform(lambda s: holm(s.to_numpy()))
    return out


# ============================================================ ablations
def summarise_run(df: pd.DataFrame) -> dict:
    out = {}
    clean = df[df["group"] == "clean"]
    for o in CFG.OVERLAPS:
        d = df[df["overlap"] == o]
        out[f"infl_p_{o}"] = d["infl_p"].mean()
        out[f"infl_acc_pts_{o}"] = 100 * d["infl_acc"].mean()
        x16 = df[df["group"] == f"{o}_x16"]
        y = np.r_[np.ones(len(x16)), np.zeros(len(clean))]
        for det in ["det_minkpp", "det_ppl"]:
            out[f"auroc_{DETECTORS[det]}_{o}_x16"] = roc_auc_score(y, np.r_[x16[det], clean[det]])
    cont = df[df["dose"] > 0]
    out["mean_within_cell_rho_minkpp"] = np.nanmean(
        [spearman(df.loc[df.group == g, "det_minkpp"], df.loc[df.group == g, "infl_p"])[0] for g in CFG.CELLS])
    out["pooled_rho_minkpp"] = spearman(cont["det_minkpp"], cont["infl_p"])[0]
    return out


def ablation_table(runs: dict) -> pd.DataFrame:
    rows = []
    for n in ["seed1", *ABLATIONS]:
        if n in runs:
            rows.append({"run": n, **summarise_run(runs[n])})
    return pd.DataFrame(rows)


def detector_setting_table(avg: pd.DataFrame) -> pd.DataFrame:
    """Detector-setting ablation from the main runs: k for Min-K%/Min-K%++, FSD base score."""
    alt = {"det_mink_10": "Min-K% k=10", "det_mink": "Min-K% k=20", "det_mink_50": "Min-K% k=50",
           "det_minkpp_10": "Min-K%++ k=10", "det_minkpp": "Min-K%++ k=20", "det_minkpp_50": "Min-K%++ k=50",
           "det_fsd": "FSD on perplexity", "det_fsd_mink": "FSD on Min-K%"}
    t = detection_table(avg, detectors=alt)
    return t.pivot_table(index="detector", columns="group", values="auroc")[CFG.CELLS].reset_index()


def control_table(runs: dict) -> pd.DataFrame:
    rows = []
    for n in [*MAIN_RUNS, "control"]:
        if n in runs:
            df = runs[n]
            rows.append({"run": n, "mean_delta_p_clean_items": df.loc[df.group == "clean", "delta_p"].mean(),
                         "mean_delta_acc_pts_clean_items": 100 * df.loc[df.group == "clean", "delta_acc"].mean(),
                         "mean_delta_p_all_test_items": df["delta_p"].mean()})
    return pd.DataFrame(rows)


# ============================================================ error analysis
def error_analysis(avg: pd.DataFrame, det: str = "det_minkpp", thr: float = INFL_THRESHOLD):
    clean = avg[avg["group"] == "clean"]
    cut = np.quantile(clean[det], 0.95)          # flag = above clean items' 95th percentile (5% FPR)
    d = avg[avg["dose"] > 0].copy()
    d["flagged"] = d[det] > cut
    d["inflated"] = d["infl_p"] >= thr
    d["outcome"] = np.select([d.flagged & d.inflated, d.flagged & ~d.inflated, ~d.flagged & d.inflated],
                             ["hit", "false alarm", "miss"], "neither")
    q25 = d["base_loss"].quantile(0.25)
    d["mem_type"] = np.select([(d["overlap"] == "exact") & (d["dose"] == 16), d["base_loss"] <= q25],
                              ["recitation", "reconstruction"], "recollection")
    counts = d.groupby(["outcome", "mem_type"]).size().unstack(fill_value=0)
    by_overlap = d.groupby(["overlap", "outcome"]).size().unstack(fill_value=0)
    return d, counts, by_overlap


def error_samples(err: pd.DataFrame, items: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    cols = ["item_id", "question", "near_dup", "paraphrase", "choices", "answer_idx"]
    out = []
    for g in ["hit", "false alarm", "miss"]:
        s = err[err["outcome"] == g]
        s = s.sample(n=min(n, len(s)), random_state=0)
        out.append(s.merge(items[cols], on="item_id")
                   [["outcome", "group", "mem_type", "infl_p", "det_minkpp", "base_p", *cols[1:]]])
    df = pd.concat(out)
    df["notes (manual)"] = ""
    return df


# ============================================================ comparison with published work
def literature_table(avg: pd.DataFrame, det_tab: pd.DataFrame, cell_tab: pd.DataFrame, abl: pd.DataFrame) -> pd.DataFrame:
    cont = avg[avg["dose"] > 0]
    corr = cont[list(DETECTORS)].corr(method="spearman").to_numpy()
    off = corr[np.triu_indices(len(DETECTORS), 1)]
    a = det_tab.pivot_table(index="group", columns="detector", values="auroc")
    best_exact = a.loc["exact_x16"].max()
    best_para = a.loc["paraphrase_x16"].max()
    ci = cell_tab.set_index("group")["infl_p"]
    ratio16 = ci["paraphrase_x16"] / ci["exact_x16"] if ci["exact_x16"] > 0 else np.nan   # x16: both effects clearly non-zero
    auc_ratio = (best_para - 0.5) / (best_exact - 0.5) if best_exact > 0.5 else np.nan
    q = abl.set_index("run")
    pair = (q.loc["qonly", "infl_p_exact"] / q.loc["seed1", "infl_p_exact"]) if {"qonly", "seed1"} <= set(q.index) else np.nan
    rows = [
        ["Min-K%++ [4]", "Membership AUROC of Min-K%++ on Pile-based MIMIR",
         "about 0.52-0.59 (average)", f"{a['Min-K%++'].min():.2f}-{a['Min-K%++'].max():.2f} across our 9 cells",
         "Pretraining membership on the Pile vs fine-tuning contamination on ARC-Easy; different model sizes."],
        ["FSD [5]", "AUROC after FSD on Pile data",
         "0.50-0.52 before, 0.57-0.63 after", f"{a['FSD'].min():.2f}-{a['FSD'].max():.2f} across our 9 cells",
         "Their unseen set is a different Pile subset; ours is unseen items from the same benchmark."],
        ["Fu et al. [3]", "Membership AUROC on fine-tuning data",
         "below 0.50 to 0.994, depends on dataset", f"best detector {best_exact:.2f} (exact x16) to {best_para:.2f} (paraphrase x16)",
         "Different models (Zephyr, BioMistral) and datasets; same idea of fine-tuning-stage membership."],
        ["Samuel et al. [8]", "Agreement between detectors (Spearman)",
         "-0.20 to 0.32", f"{off.min():.2f} to {off.max():.2f} between our 5 detectors",
         "They compare dataset-level verdicts across benchmarks; we correlate item-level scores."],
        ["Shilov et al. [12]", "Paraphrase vs exact copy (how much a paraphrase counts)",
         "0.13-0.34 of an exact copy (membership)",
         f"at x16: paraphrase gives {ratio16:.2f} of the exact-copy inflation and {auc_ratio:.2f} of its detection signal (AUROC above 0.5)",
         "They measure membership signal; we measure score inflation and detection; different models."],
        ["Kocyigit et al. [1], Yang et al. [2]", "Is the answer needed for inflation?",
         "only paired input+answer inflates scores", f"question-only / paired exact inflation = {pair:.2f}",
         "Translation and code tasks vs multiple-choice science questions."],
        ["Bordt et al. [11]", "Does inflation grow with repetitions?",
         "yes; more repetitions, more inflation", f"exact: {ci['exact_x1']:+.3f} (x1), {ci['exact_x4']:+.3f} (x4), {ci['exact_x16']:+.3f} (x16) in p_correct",
         "Pretraining from scratch at larger scale vs LoRA fine-tuning of a 410M model."],
    ]
    return pd.DataFrame(rows, columns=["paper", "quantity", "reported", "ours", "limits_on_fair_comparison"])


# ============================================================ paraphrase-quality sensitivity
def paraphrase_sensitivity(avg: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Paraphrase cells recomputed with only paraphrases that passed the filter (sim >= 0.85, no shared 8-gram)."""
    ok = items.set_index("item_id")["para_passed"]
    clean = avg[avg["group"] == "clean"]
    rows = []
    for d in CFG.DOSES:
        g = f"paraphrase_x{d}"
        cell = avg[avg["group"] == g]
        for label, sub in [("all paraphrases", cell), ("passed filter only", cell[cell["item_id"].map(ok) == 1])]:
            y = np.r_[np.ones(len(sub)), np.zeros(len(clean))]
            r = {"group": g, "subset": label, "n_items": len(sub), "infl_p": sub["infl_p"].mean(),
                 "infl_acc_pts": 100 * sub["infl_acc"].mean()}
            for det, lab in DETECTORS.items():
                r[f"auroc_{lab}"] = roc_auc_score(y, np.r_[sub[det], clean[det]]) if len(sub) > 5 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


# ============================================================ everything
def run_all(root: Path = CFG.ROOT, out_dir: Path | None = None) -> dict:
    root = Path(root)
    out_dir = Path(out_dir or root / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    avg = seed_average(runs)
    items = pd.read_csv(root / "data" / "items.csv")
    res = {"runs": runs, "avg": avg, "items": items}
    res["cell"] = cell_inflation(avg, runs)
    res["detection"] = detection_table(avg)
    res["delong"] = delong_table(avg, res["detection"])
    res["agreement"] = agreement_table(avg)
    res["partial"] = partial_spearman_dose(avg)
    res["predictor"] = predictor_table(avg)
    res["mixed"], mixed_txt = mixed_model(runs)
    res["mcnemar"] = mcnemar_table(runs)
    res["ablation"] = ablation_table(runs)
    res["detector_settings"] = detector_setting_table(avg)
    res["control"] = control_table(runs)
    res["err"], res["err_counts"], res["err_by_overlap"] = error_analysis(avg)
    res["err_samples"] = error_samples(res["err"], items)
    if "para_passed" in items:
        res["para_sensitivity"] = paraphrase_sensitivity(avg, items)
        res["para_sensitivity"].to_csv(out_dir / "paraphrase_sensitivity.csv", index=False)
    res["literature"] = literature_table(avg, res["detection"], res["cell"], res["ablation"])
    res["inter_detector"] = avg[avg["dose"] > 0][list(DETECTORS)].corr(method="spearman") \
        .rename(index=DETECTORS, columns=DETECTORS)
    for k in ["cell", "detection", "delong", "agreement", "partial", "predictor", "mixed", "mcnemar", "ablation",
              "detector_settings", "control", "err_samples", "literature"]:
        res[k].to_csv(out_dir / f"{k}.csv", index=False)
    res["err_counts"].to_csv(out_dir / "error_counts.csv")
    res["err_by_overlap"].to_csv(out_dir / "error_by_overlap.csv")
    res["inter_detector"].to_csv(out_dir / "inter_detector_spearman.csv")
    avg.to_csv(out_dir / "item_level_dataset.csv", index=False)
    (out_dir / "mixed_model_summary.txt").write_text(mixed_txt)
    logs = {p.stem: json.loads(p.read_text()) for p in (root / "results").glob("*_log.json")}
    (out_dir / "run_logs.json").write_text(json.dumps(logs, indent=2, default=str))
    res["logs"] = logs
    return res
