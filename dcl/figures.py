"""Report figures (Phase 3, requirement 4). Light theme, colour-blind-checked palette,
direct labels, one y-axis per chart."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.metrics import roc_curve

from . import config as C
from .analysis import DETECTORS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
OVERLAP_COL = dict(zip(C.OVERLAPS, CAT[:3]))
OVERLAP_LAB = {"exact": "Exact copy", "near_dup": "Near-duplicate", "paraphrase": "Paraphrase"}
OVERLAP_MARK = {"exact": "o", "near_dup": "s", "paraphrase": "^"}
DET_COL = dict(zip(DETECTORS.values(), CAT[:5]))
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f3f2ef", "#2a78d6"])
DOSE_COL = {1: "#86b6ef", 4: "#3987e5", 16: "#104281"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2,
    "figure.constrained_layout.use": True,
})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out / f"{name}.png"


def _grid(df, value):
    return df.pivot_table(index="overlap", columns="dose", values=value).reindex(index=C.OVERLAPS, columns=C.DOSES)


def _heat(ax, M, fmt, cmap, vmin=None, vmax=None, cbar_label=""):
    im = ax.imshow(M.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1]), [f"x{c}" if isinstance(c, (int, np.integer)) else str(c) for c in M.columns])
    ax.set_yticks(range(M.shape[0]), [OVERLAP_LAB.get(i, i) for i in M.index])
    ax.grid(False)
    lo, hi = np.nanmin(M.to_numpy()), np.nanmax(M.to_numpy())
    for (i, j), v in np.ndenumerate(M.to_numpy()):
        frac = (v - (vmin if vmin is not None else lo)) / max(1e-9, ((vmax if vmax is not None else hi) - (vmin if vmin is not None else lo)))
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                color="white" if frac > 0.6 else INK)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, color=INK2)
    cb.outline.set_visible(False)
    return im


# ------------------------------------------------------------------ F1-F2: RQ1 inflation
def fig_inflation_heatmap(cell, out):
    c = cell[cell["dose"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    _heat(axes[0], _grid(c, "infl_p"), "{:+.3f}", BLUES, vmin=min(0, c["infl_p"].min()), cbar_label="inflation in p(correct)")
    axes[0].set_title("Inflation in p(correct answer)")
    _heat(axes[1], _grid(c, "infl_acc_pts"), "{:+.1f}", BLUES, vmin=min(0, c["infl_acc_pts"].min()), cbar_label="accuracy points")
    axes[1].set_title("Inflation in accuracy (points)")
    for a in axes:
        a.set_xlabel("Copies seen during training (dose)")
    fig.suptitle("Figure 1. Score inflation by dose and overlap type (vs clean items, mean of 3 seeds)", x=0.01, ha="left", fontsize=11)
    return _save(fig, out, "fig01_inflation_heatmap")


def fig_inflation_dose(cell, out):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for o in C.OVERLAPS:
        d = cell[cell["overlap"] == o].sort_values("dose")
        ax.fill_between(d["dose"], d["infl_p_lo"], d["infl_p_hi"], color=OVERLAP_COL[o], alpha=0.15, lw=0)
        ax.plot(d["dose"], d["infl_p"], color=OVERLAP_COL[o], marker=OVERLAP_MARK[o], ms=7)
        ax.annotate(OVERLAP_LAB[o], (d["dose"].iloc[-1], d["infl_p"].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(C.DOSES, [f"x{d}" for d in C.DOSES])
    ax.set_xlim(0.8, 40)
    ax.set_xlabel("Copies seen during training (log scale)")
    ax.set_ylabel("Inflation in p(correct)")
    ax.set_title("Figure 2. Inflation grows with dose; bands are 95% bootstrap CIs", loc="left")
    return _save(fig, out, "fig02_inflation_vs_dose")


# ------------------------------------------------------------------ F3-F4: RQ2 detection
def fig_auroc_heatmap(det, out):
    piv = det.pivot_table(index="detector", columns="group", values="auroc").reindex(index=list(DETECTORS.values()), columns=C.CELLS)
    piv.columns = [g.replace("near_dup", "near-dup").replace("_x", " x") for g in piv.columns]
    fig, ax = plt.subplots(figsize=(10.5, 3.3))
    im = ax.imshow(piv.to_numpy(), cmap=DIVERGE, norm=TwoSlopeNorm(vcenter=0.5, vmin=0.3, vmax=1.0), aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(piv.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.85 else INK)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("AUROC (0.5 = chance)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Figure 3. Detection AUROC: contaminated items of each cell vs clean items", loc="left")
    return _save(fig, out, "fig03_auroc_heatmap")


def fig_roc_curves(avg, out):
    clean = avg[avg["group"] == "clean"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    for ax, o in zip(axes, C.OVERLAPS):
        d = avg[avg["group"] == f"{o}_x16"]
        y = np.r_[np.ones(len(d)), np.zeros(len(clean))]
        for col, lab in DETECTORS.items():
            f, t, _ = roc_curve(y, np.r_[d[col], clean[col]])
            ax.plot(f, t, color=DET_COL[lab], lw=1.8, label=lab)
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
        ax.set_title(f"{OVERLAP_LAB[o]}, x16", loc="left")
        ax.set_xlabel("False positive rate")
        ax.set_aspect("equal")
    axes[0].set_ylabel("True positive rate")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Figure 4. ROC curves at the highest dose: detection weakens from exact copies to paraphrases",
                 x=0.01, ha="left", fontsize=11)
    return _save(fig, out, "fig04_roc_curves")


# ------------------------------------------------------------------ F5-F7: RQ3 agreement
def fig_scatter(avg, out, det="det_minkpp"):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    lab = DETECTORS[det]
    for ax, o in zip(axes, C.OVERLAPS):
        for dose in C.DOSES:
            d = avg[avg["group"] == f"{o}_x{dose}"]
            ax.scatter(d[det], d["infl_p"], s=14, color=DOSE_COL[dose], alpha=0.75, edgecolor=SURFACE,
                       linewidth=0.6, label=f"x{dose}")
        cl = avg[avg["group"] == "clean"]
        ax.scatter(cl[det], cl["infl_p"], s=8, color="#bdbcb6", alpha=0.6, label="clean", zorder=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(OVERLAP_LAB[o], loc="left")
        ax.set_xlabel(f"{lab} score (higher = 'seen')")
    axes[0].set_ylabel("Item inflation in p(correct)")
    axes[-1].legend(title="dose", fontsize=8, title_fontsize=8, loc="upper left")
    fig.suptitle(f"Figure 5. Item-level detector score vs score inflation ({lab})", x=0.01, ha="left", fontsize=11)
    return _save(fig, out, f"fig05_scatter_{det.replace('det_', '')}")


def fig_forest(agr, out):
    w = agr[agr["scope"] == "within cell"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True, sharey=True)
    dets = list(DETECTORS.values())
    for ax, o in zip(axes, C.OVERLAPS):
        for k, dose in enumerate(C.DOSES):
            g = f"{o}_x{dose}"
            for j, dl in enumerate(dets):
                r = w[(w["group"] == g) & (w["detector"] == dl)].iloc[0]
                y = k * (len(dets) + 1.5) + j
                ax.plot([r["rho_lo"], r["rho_hi"]], [y, y], color=DET_COL[dl], lw=2)
                ax.plot(r["rho"], y, "o", color=DET_COL[dl], ms=5, mec=SURFACE, mew=0.8)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks([k * (len(dets) + 1.5) + 2 for k in range(3)], [f"x{d}" for d in C.DOSES])
        ax.set_title(OVERLAP_LAB[o], loc="left")
        ax.set_xlabel("Spearman ρ (score vs inflation)")
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    handles = [plt.Line2D([], [], color=DET_COL[d], marker="o", lw=2, label=d) for d in dets]
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle("Figure 6. Within-cell correlation between detector score and inflation (95% bootstrap CI)",
                 x=0.01, ha="left", fontsize=11)
    return _save(fig, out, "fig06_within_cell_rho")


def fig_predictor(pred, out):
    keep = ["All 5 detectors + item features", "All 5 detectors", *[f"{l} only" for l in DETECTORS.values()],
            "Dose x overlap only (knows true exposure)", "Difficulty only (base p_correct)", "Random scores"]
    d = pred.set_index("predictor").loc[[k for k in keep if k in set(pred["predictor"])]].reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    reference = {"Dose x overlap only (knows true exposure)", "Difficulty only (base p_correct)", "Random scores"}
    col = ["#a3a29b" if p in reference else "#2a78d6" for p in d["predictor"]]
    ax.barh(range(len(d)), d["cv_r2"], color=col, height=0.62)
    for i, v in enumerate(d["cv_r2"]):
        ax.text(max(v, 0) + 0.01, i, f"{v:.2f}", va="center", ha="left", fontsize=8.5)
    ax.set_yticks(range(len(d)), d["predictor"])
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Cross-validated R² for predicting item inflation")
    ax.grid(axis="y", visible=False)
    ax.set_title("Figure 7. How well detector scores predict inflation, vs reference predictors (grey)", loc="left")
    return _save(fig, out, "fig07_combined_predictor")


# ------------------------------------------------------------------ F8: ablations
ABL_LAB = {"seed1": "Main (seed 1)", "qonly": "Question only", "dilution": "Half filler",
           "rank8": "LoRA rank 8", "fullft160m": "Full FT 160M", "rank64": "LoRA rank 64", "filler10m": "Double filler"}


def fig_ablation(abl, out):
    runs = [r for r in ABL_LAB if r in set(abl["run"])]
    a = abl.set_index("run").loc[runs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    x = np.arange(len(runs))
    w = 0.26
    for k, o in enumerate(C.OVERLAPS):
        axes[0].bar(x + (k - 1) * w, a[f"infl_p_{o}"], w - 0.02, color=OVERLAP_COL[o], label=OVERLAP_LAB[o])
        axes[1].bar(x + (k - 1) * w, a[f"auroc_Min-K%++_{o}_x16"], w - 0.02, color=OVERLAP_COL[o], label=OVERLAP_LAB[o])
    axes[0].set_ylabel("Mean inflation in p(correct)")
    axes[0].set_title("Inflation by overlap type", loc="left")
    axes[1].set_ylabel("Min-K%++ AUROC at x16")
    axes[1].axhline(0.5, color=INK2, lw=1, ls="--")
    axes[1].set_ylim(0.3, 1.0)
    axes[1].set_title("Detection by overlap type", loc="left")
    for ax in axes:
        ax.set_xticks(x, [ABL_LAB[r] for r in runs], rotation=20, ha="right")
        ax.grid(axis="x", visible=False)
    axes[0].axhline(0, color=INK2, lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 8. Ablations: each run changes one component of the main design (one seed)", x=0.01, ha="left", fontsize=11)
    return _save(fig, out, "fig08_ablations")


# ------------------------------------------------------------------ F9: error analysis
def fig_errors(counts, by_overlap, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    order = [o for o in ["hit", "false alarm", "miss", "neither"] if o in counts.index]
    mt = [m for m in ["recitation", "reconstruction", "recollection"] if m in counts.columns]
    left = np.zeros(len(order))
    for k, m in enumerate(mt):
        v = counts.loc[order, m].to_numpy()
        axes[0].barh(order, v, left=left, color=CAT[k + 3] if k else CAT[0], label=m, height=0.6, edgecolor=SURFACE, linewidth=2)
        left += v
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Contaminated items")
    axes[0].set_title("Outcome by memorization type", loc="left")
    axes[0].legend(fontsize=8)
    bo = by_overlap.reindex(index=C.OVERLAPS, columns=order, fill_value=0)
    share = bo.div(bo.sum(1), axis=0)
    left = np.zeros(len(share))
    shades = {"hit": "#2a78d6", "false alarm": "#eda100", "miss": "#eb6834", "neither": "#d4d3cd"}
    for o in order:
        axes[1].barh([OVERLAP_LAB[i] for i in share.index], share[o], left=left, color=shades[o], label=o,
                     height=0.6, edgecolor=SURFACE, linewidth=2)
        left += share[o].to_numpy()
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Share of items")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Outcome by overlap type", loc="left")
    axes[1].legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 9. Error analysis with Min-K%++ (flag = above clean items' 95th percentile; inflated = +0.10 p(correct))",
                 x=0.01, ha="left", fontsize=11)
    return _save(fig, out, "fig09_error_analysis")


# ------------------------------------------------------------------ F10-F11: comparison with literature
def fig_literature(det, inter, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ref = [("Fu et al. [3]: pretraining data", 0.49, 0.53), ("Min-K%++ [4]: MIMIR", 0.52, 0.59),
           ("FSD [5]: Pile, after FSD", 0.57, 0.63), ("Fu et al. [3]: fine-tuning data", 0.41, 0.994)]
    for i, (lab, lo, hi) in enumerate(ref):
        ax.plot([lo, hi], [i, i], color="#a3a29b", lw=6, solid_capstyle="round")
    ours = det.groupby("overlap")["auroc"].agg(["min", "max"]).reindex(C.OVERLAPS)
    for k, o in enumerate(C.OVERLAPS):
        i = len(ref) + k
        ax.plot([ours.loc[o, "min"], ours.loc[o, "max"]], [i, i], color=OVERLAP_COL[o], lw=6, solid_capstyle="round")
    ax.set_yticks(range(len(ref) + 3), [r[0] for r in ref] + [f"Ours: {OVERLAP_LAB[o]} (all detectors, doses)" for o in C.OVERLAPS])
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.set_xlabel("AUROC range")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("Reported AUROC ranges (grey) vs ours", loc="left")
    ax = axes[1]
    M = inter.to_numpy()
    im = ax.imshow(M, cmap=DIVERGE, vmin=-1, vmax=1)
    ax.set_xticks(range(len(inter)), inter.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(inter)), inter.index)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if abs(v) > 0.7 else INK)
    ax.set_title("Agreement between detectors (Spearman)\nSamuel et al. [8] report -0.20 to 0.32", loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    fig.suptitle("Figure 10. Comparison with published results (different settings; see Table of limits)", x=0.01, ha="left", fontsize=11)
    return _save(fig, out, "fig10_literature_comparison")


def fig_control(ctrl, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    lab = {"seed1": "Seed 1 (clean items)", "seed2": "Seed 2 (clean items)", "seed3": "Seed 3 (clean items)",
           "control": "Format-only control"}
    d = ctrl.set_index("run")
    names = [r for r in lab if r in d.index]
    v = d.loc[names, "mean_delta_acc_pts_clean_items"]
    ax.barh([lab[n] for n in names], v, color=["#eb6834" if n == "control" else "#86b6ef" for n in names], height=0.55)
    for i, x in enumerate(v):
        ax.text(x, i, f" {x:+.1f}", va="center", fontsize=9)
    ax.axvline(0, color=INK2, lw=1)
    ax.invert_yaxis()
    ax.set_xlabel("Change in accuracy on never-seen items (points)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Figure 11. Format learning: gain on items the model never saw", loc="left")
    return _save(fig, out, "fig11_format_control")


def fig_training_loss(root, out):
    files = sorted(Path(root, "results").glob("*_trainloss.csv"))
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for k, f in enumerate(files):
        d = pd.read_csv(f)
        ax.plot(d["step"], d["loss"].rolling(10, min_periods=1).mean(), lw=1.5, color=CAT[k % 6] if k < 6 else INK2,
                label=f.stem.replace("_trainloss", ""))
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel("Training loss (10-step mean)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Figure 12. Training loss of every run", loc="left")
    return _save(fig, out, "fig12_training_loss")


def make_all(res: dict, root: Path = C.ROOT, out: Path | None = None) -> list[Path]:
    out = Path(out or Path(root) / "figures")
    paths = [fig_inflation_heatmap(res["cell"], out), fig_inflation_dose(res["cell"], out),
             fig_auroc_heatmap(res["detection"], out), fig_roc_curves(res["avg"], out),
             fig_scatter(res["avg"], out, "det_minkpp"), fig_scatter(res["avg"], out, "det_ppl"),
             fig_forest(res["agreement"], out), fig_predictor(res["predictor"], out),
             fig_ablation(res["ablation"], out), fig_errors(res["err_counts"], res["err_by_overlap"], out),
             fig_literature(res["detection"], res["inter_detector"], out), fig_control(res["control"], out),
             fig_training_loss(root, out)]
    return [p for p in paths if p]
