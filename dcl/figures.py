"""Report figures (Phase 3, requirement 4; Milestone 4 paper).

Design rules (checked with the dataviz palette validator):
- Colour always means the same thing: overlap type = categorical slots 1-3 (blue, orange, aqua);
  detectors = slots 4-8 (yellow, magenta, green, violet, red); dose = lightness of the overlap hue;
  neutral greys for reference predictors and context; status red only for 'missed' items.
- Figures are drawn at print size (full width 7.16 in, column 3.5 in) so text prints at 7-8.5 pt.
- One y-axis per chart, hairline grid, direct labels where they help, uncertainty shown.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from scipy import stats
from sklearn.metrics import roc_curve

from . import config as C
from .analysis import DETECTORS

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OVERLAP_COL = dict(zip(C.OVERLAPS, CAT[:3]))
OVERLAP_LAB = {"exact": "Exact copy", "near_dup": "Near-duplicate", "paraphrase": "Paraphrase"}
OVERLAP_MARK = {"exact": "o", "near_dup": "s", "paraphrase": "^"}
# Detector palette validated for colour-vision deficiency (all pairs, light surface); every detector also
# has its own marker and line style, so identity never depends on colour alone.
DET_COL = dict(zip(DETECTORS.values(), ["#d99a00", "#e87ba4", "#007a5e", "#6250d6", "#8a3b12"]))
DET_MARK = dict(zip(DETECTORS.values(), ["o", "s", "^", "D", "v"]))
DET_LS = dict(zip(DETECTORS.values(), ["-", (0, (4, 1.5)), (0, (1, 1)), (0, (5, 1.5, 1, 1.5)), (0, (2, 1, 2, 3))]))
OVERLAP_HATCH = {"exact": "", "near_dup": "////", "paraphrase": "...."}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
REF_GREY, CTX_GREY, LIGHT_GREY = "#a3a29b", "#c3c2b7", "#d9d8d2"
CRITICAL = "#d03b3b"
FULL, COL = 7.16, 3.5


def _tint(hexcol, f):
    """Mix a colour with the surface: f=0 -> colour, f=1 -> surface (one-hue ordinal ramp)."""
    a, b = np.array(to_rgb(hexcol)), np.array(to_rgb(SURFACE))
    return tuple(a * (1 - f) + b * f)


DOSE_TINT = {1: 0.5, 4: 0.25, 16: 0.0}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "hatch.linewidth": 0.6, "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "axes.axisbelow": True, "lines.linewidth": 1.6, "figure.constrained_layout.use": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _suptitle(fig, text):
    fig.suptitle(text, x=0.005, ha="left", fontsize=8.5, color=INK)


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out / f"{name}.png"


def _grid(df, value):
    return df.pivot_table(index="overlap", columns="dose", values=value).reindex(index=C.OVERLAPS, columns=C.DOSES)


def _cbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(label, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    return cb


def _heat(fig, ax, M, fmt, cmap, vmin, vmax, cbar_label):
    im = ax.imshow(M.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(M.shape[1]), [f"×{c}" for c in M.columns])
    ax.set_yticks(range(M.shape[0]), [OVERLAP_LAB.get(i, i) for i in M.index])
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for (i, j), v in np.ndenumerate(M.to_numpy()):
        frac = (v - vmin) / max(1e-9, vmax - vmin)
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5, color="white" if abs(frac - 0.5) > 0.36 else INK)
    _cbar(fig, im, ax, cbar_label)


# ------------------------------------------------------------------ RQ1 inflation
def fig_inflation_heatmap(cell, out):
    c = cell[cell["dose"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.1))
    for ax, col, fmt, lab, title in [(axes[0], "infl_p", "{:+.3f}", "p(correct)", "Inflation in p(correct)"),
                                     (axes[1], "infl_acc_pts", "{:+.1f}", "accuracy points", "Inflation in accuracy points")]:
        M = _grid(c, col)
        m = float(np.abs(M.to_numpy()).max())
        _heat(fig, ax, M, fmt, DIVERGE, -m, m, lab)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Copies inserted (dose)")
    return _save(fig, out, "fig01_inflation_heatmap")


def fig_inflation_dose(cell, out):
    """Point estimates with 95% CIs (joint bootstrap), dodged so intervals do not overlap."""
    fig, ax = plt.subplots(figsize=(COL, 2.55))
    x0 = {1: 0, 4: 1, 16: 2}
    for k, o in enumerate(C.OVERLAPS):
        d = cell[cell["overlap"] == o].sort_values("dose")
        x = np.array([x0[v] for v in d["dose"]]) + (k - 1) * 0.16
        ax.vlines(x, d["infl_p_lo"], d["infl_p_hi"], color=OVERLAP_COL[o], lw=1.4)
        ax.plot(x, d["infl_p"], color=OVERLAP_COL[o], marker=OVERLAP_MARK[o], ms=5, lw=1.2,
                mec=SURFACE, mew=0.8, label=OVERLAP_LAB[o])
        ax.annotate(f"{OVERLAP_LAB[o]} {d['infl_p'].iloc[-1]:+.3f}", (x[-1], d["infl_p"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=6.8, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks([0, 1, 2], ["×1", "×4", "×16"])
    ax.set_xlim(-0.35, 3.0)
    ax.set_xlabel("Copies inserted (dose, log spacing)")
    ax.set_ylabel("Inflation: Δ p(correct) vs clean items")
    ax.grid(axis="x", visible=False)
    return _save(fig, out, "fig02_inflation_vs_dose")


# ------------------------------------------------------------------ RQ2 detection
def fig_auroc_heatmap(det, out):
    """Diverging scale symmetric about chance (0.5); colour steps are equal on both sides."""
    dets = list(DETECTORS.values())
    piv = det.pivot_table(index="detector", columns="group", values="auroc").reindex(index=dets, columns=C.CELLS)
    fig, ax = plt.subplots(figsize=(5.9, 2.1))
    M = piv.to_numpy()
    im = ax.imshow(M, cmap=DIVERGE, vmin=0.1, vmax=0.9, aspect="auto")
    ax.set_xticks(range(9), ["×1", "×4", "×16"] * 3)
    ax.set_yticks(range(len(dets)), dets)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.79 else INK)
    for b in (2.5, 5.5):
        ax.axvline(b, color=SURFACE, lw=3)
    for k, o in enumerate(C.OVERLAPS):
        ax.text(3 * k + 1, -0.75, OVERLAP_LAB[o], ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=INK)
    ax.set_ylim(len(dets) - 0.5, -1.05)
    ax.set_xlabel("Copies inserted (dose) within each overlap type")
    ax.set_ylabel("Detector")
    cb = _cbar(fig, im, ax, "AUROC (0.5 = chance)")
    cb.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9])
    return _save(fig, out, "fig03_auroc_heatmap")


def fig_roc_curves(avg, out):
    clean = avg[avg["group"] == "clean"]
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.35), sharey=True)
    for ax, o in zip(axes, C.OVERLAPS):
        d = avg[avg["group"] == f"{o}_x16"]
        y = np.r_[np.ones(len(d)), np.zeros(len(clean))]
        for col, lab in DETECTORS.items():
            f, t, _ = roc_curve(y, np.r_[d[col], clean[col]])
            ax.plot(f, t, color=DET_COL[lab], ls=DET_LS[lab], lw=1.3, label=lab)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(f"{OVERLAP_LAB[o]}, ×16", loc="left")
        ax.set_xlabel("False positive rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_box_aspect(1)
    axes[0].set_ylabel("True positive rate")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, handlelength=2.2)
    return _save(fig, out, "fig04_roc_curves")


# ------------------------------------------------------------------ RQ3 item-level link
def fig_scatter(avg, out, det="det_minkpp"):
    """Hue = overlap type, lightness = dose; ink line = mean inflation in score deciles (all doses)."""
    from matplotlib.legend_handler import HandlerTuple
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.6), sharey=True)
    lab = DETECTORS[det]
    shown = avg[avg["group"].isin(["clean"] + [f"{o}_x{d}" for o in C.OVERLAPS for d in C.DOSES])]
    lo, hi = shown[det].quantile([0.005, 0.995])
    pad = 0.05 * (hi - lo)
    cl = avg[avg["group"] == "clean"]
    for ax, o in zip(axes, C.OVERLAPS):
        ax.scatter(cl[det], cl["infl_p"], s=4, color=REF_GREY, alpha=0.35, edgecolor="none", zorder=1)
        cont = avg[avg["overlap"] == o]
        for dose in C.DOSES:
            d = cont[cont["dose"] == dose]
            ax.scatter(d[det], d["infl_p"], s=5, color=_tint(OVERLAP_COL[o], DOSE_TINT[dose]), alpha=0.6,
                       edgecolor="none", zorder=2 + dose / 100)
        q = pd.qcut(cont[det], 10, labels=False, duplicates="drop")
        b = cont.groupby(q).agg(x=(det, "median"), y=("infl_p", "mean"))
        ax.plot(b["x"], b["y"], color=INK, lw=1.4, marker="o", ms=2.8, zorder=5)
        ax.axhline(0, color=INK2, lw=0.7, zorder=0)
        within = np.mean([stats.spearmanr(g[det], g["infl_p"]).statistic for _, g in cont.groupby("dose")])
        ax.set_title(OVERLAP_LAB[o], loc="left")
        ax.text(0.02, 0.97, f"mean within-cell ρ = {within:.2f}", transform=ax.transAxes, va="top", fontsize=7,
                color=INK, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85), zorder=6)
        ax.set_xlabel(f"{lab} score (higher = 'seen')")
        ax.set_xlim(lo - pad, hi + pad)
    axes[0].set_ylabel("Item inflation, Δ p(correct)")
    mk = lambda c: plt.Line2D([], [], ls="", marker="o", ms=4, color=c)
    h = [mk(REF_GREY)] + [tuple(mk(_tint(OVERLAP_COL[o], DOSE_TINT[d])) for o in C.OVERLAPS) for d in C.DOSES]
    h += [plt.Line2D([], [], color=INK, lw=1.4, marker="o", ms=2.8)]
    fig.legend(h, ["clean (never inserted)", "×1", "×4", "×16", "decile mean"], loc="outside lower center", ncol=5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)}, handletextpad=0.4, columnspacing=1.4)
    return _save(fig, out, f"fig05_scatter_{det.replace('det_', '')}")


def fig_forest(agr, out):
    w = agr[agr["scope"] == "within cell"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(5.9, 3.3), sharex=True, sharey=True)
    dets = list(DETECTORS.values())
    step = len(dets) + 1.5
    for ax, o in zip(axes, C.OVERLAPS):
        for k, dose in enumerate(C.DOSES):
            g = f"{o}_x{dose}"
            sub = w[w["group"] == g]
            for j, dl in enumerate(dets):
                r = sub[sub["detector"] == dl].iloc[0]
                y = k * step + j
                ax.plot([r["rho_lo"], r["rho_hi"]], [y, y], color=DET_COL[dl], lw=1.3)
                ax.plot(r["rho"], y, DET_MARK[dl], color=DET_COL[dl], ms=3.8, mec=SURFACE, mew=0.6)
            m = sub["rho"].mean()
            ax.plot([m, m], [k * step - 0.6, k * step + len(dets) - 0.4], color=INK, lw=1.0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(OVERLAP_LAB[o], loc="left")
        ax.set_xlabel("Spearman ρ")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    ticks = [k * step + j for k in range(3) for j in range(len(dets))]
    axes[0].set_yticks(ticks, [f"{d}" for _ in C.DOSES for d in dets], fontsize=6.5)
    axes[0].invert_yaxis()
    for k, dose in enumerate(C.DOSES):
        axes[-1].text(1.03, k * step + 2, f"×{dose}", transform=axes[-1].get_yaxis_transform(), va="center",
                      fontsize=7.5, fontweight="bold", color=INK)
    fig.legend(handles=[plt.Line2D([], [], color=INK, lw=1.0, label="mean of the 5 detectors in that cell")],
               loc="outside lower center", fontsize=6.5, handlelength=1.2)
    return _save(fig, out, "fig06_within_cell_rho")


def fig_cell_vs_item(cell, det, agr, out, det_name="Min-K%++"):
    """Key contrast: across cells detection tracks inflation; within cells it barely does."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.55), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    a = det[det["detector"] == det_name].set_index("group")["auroc"]
    c = cell[cell["dose"] > 0].set_index("group")
    for o in C.OVERLAPS:
        g = [f"{o}_x{d}" for d in C.DOSES]
        ax.plot(c.loc[g, "infl_p"], a.loc[g], color=OVERLAP_COL[o], lw=1.1, zorder=1)
        ax.scatter(c.loc[g, "infl_p"], a.loc[g], s=26, color=OVERLAP_COL[o], marker=OVERLAP_MARK[o],
                   edgecolor=SURFACE, linewidth=0.8, zorder=3, label=OVERLAP_LAB[o])
        for gg, d in ((g[-1], 16),):
            ax.annotate(f"×{d}", (c.loc[gg, "infl_p"], a.loc[gg]), xytext=(4, 3) if d == 16 else (-4, 4),
                        ha="left" if d == 16 else "right", textcoords="offset points", fontsize=6.5, color=INK2)
    rho_cells = stats.spearmanr(c.loc[a.index.intersection(c.index), "infl_p"],
                                a.loc[a.index.intersection(c.index)]).statistic
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.99, 0.5, "chance", transform=ax.get_yaxis_transform(), va="bottom", ha="right",
            fontsize=6.5, color=MUTED)
    ax.set_xlabel("Cell inflation, Δ p(correct)")
    ax.set_ylabel(f"{det_name} AUROC")
    ax.set_title(f"Across the 9 cells: ρ = {rho_cells:.2f}", loc="left")
    ax.legend(loc="lower right", handletextpad=0.2, title="each line: ×1 → ×4 → ×16", title_fontsize=6.5)
    ax = axes[1]
    w = agr[(agr["scope"] == "within cell") & (agr["detector"] == det_name)].set_index("group")
    for k, o in enumerate(C.OVERLAPS):
        for j, d in enumerate(C.DOSES):
            r = w.loc[f"{o}_x{d}"]
            y = k * 3.8 + j
            ax.plot([r["rho_lo"], r["rho_hi"]], [y, y], color=_tint(OVERLAP_COL[o], DOSE_TINT[d]), lw=1.6)
            ax.plot(r["rho"], y, OVERLAP_MARK[o], color=_tint(OVERLAP_COL[o], DOSE_TINT[d]), ms=4,
                    mec=SURFACE, mew=0.6)
    ax.set_yticks([k * 3.8 + j for k in range(3) for j in range(3)],
                  [f"{OVERLAP_LAB[o].split()[0].split('-')[0]} ×{d}" for o in C.OVERLAPS for d in C.DOSES])
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Spearman ρ, score vs item inflation")
    ax.set_title(f"Within each cell: mean ρ = {w['rho'].mean():.2f}", loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return _save(fig, out, "fig13_cell_vs_item")


def fig_predictor(pred, out, extended=None):
    """Detector-based models in dark ink, reference predictors in grey; matches Table V labels."""
    rows = [("Random scores", "Random scores", True), ("FSD only", "FSD only", False),
            ("Min-K%++ only", "Min-K%++ only", False), ("Zlib only", "Zlib only", False),
            ("Dose x overlap only (knows true exposure)", "True dose × overlap", True),
            ("All 5 detectors", "All 5 detectors", False),
            ("Difficulty only (base p_correct)", "Difficulty only", True)]
    vals = pred.set_index("predictor")["cv_r2"]
    data = [(lab, float(vals[k]), ref) for k, lab, ref in rows if k in vals.index]
    if extended is not None:
        e = extended.set_index("predictor")["cv_r2_seed0"]
        for k, lab, ref in [("Difficulty + base loss", "Difficulty + base loss", True),
                            ("True exposure (dose x overlap) + difficulty + base loss", "Exposure + difficulty + loss", True),
                            ("All 5 detectors + difficulty + base loss", "5 detectors + difficulty + loss", False)]:
            if k in e.index:
                data.append((lab, float(e[k]), ref))
    else:
        data.append(("5 detectors + item features", float(vals["All 5 detectors + item features"]), False))
    data.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    y = np.arange(len(data))
    ax.barh(y, [max(v, 0) for _, v, _ in data], color=[REF_GREY if r else INK2 for _, _, r in data], height=0.62)
    for i, (_, v, _) in enumerate(data):
        ax.text(max(v, 0) + 0.006, i, f"{abs(v):.2f}" if abs(v) < 0.005 else f"{v:.2f}", va="center", fontsize=6.8)
    ax.set_yticks(y, [d[0] for d in data])
    ax.set_xlabel("Cross-validated R² (item inflation)")
    ax.set_xlim(0, 0.42)
    ax.grid(False)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    for i, (_, v, r) in enumerate(data):
        if abs(v) < 0.005:
            ax.plot([0, 0], [i - 0.31, i + 0.31], color=REF_GREY if r else INK2, lw=2)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=INK2, label="uses detector scores"),
                       plt.Rectangle((0, 0), 1, 1, color=REF_GREY, label="reference predictor")],
              loc="lower right", handlelength=1, handleheight=0.8)
    return _save(fig, out, "fig07_combined_predictor")


# ------------------------------------------------------------------ ablations
ABL_LAB = {"seed1": "Main (seed 1)", "qonly": "Question only", "dilution": "Half filler",
           "rank8": "LoRA rank 8", "fullft160m": "Full FT 160M", "rank64": "LoRA rank 64", "filler10m": "Double filler"}


def fig_ablation(abl, out):
    runs = [r for r in ABL_LAB if r in set(abl["run"])]
    a = abl.set_index("run").loc[runs]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    x = np.arange(len(runs))
    w = 0.26
    for k, o in enumerate(C.OVERLAPS):
        axes[0].bar(x + (k - 1) * w, a[f"infl_acc_pts_{o}"], w - 0.03, color=OVERLAP_COL[o], label=OVERLAP_LAB[o],
                    hatch=OVERLAP_HATCH[o], edgecolor=SURFACE, linewidth=0)
        axes[1].bar(x + (k - 1) * w, a[f"auroc_Min-K%++_{o}_x16"] - 0.5, w - 0.03, bottom=0.5,
                    color=OVERLAP_COL[o], label=OVERLAP_LAB[o], hatch=OVERLAP_HATCH[o], edgecolor=SURFACE, linewidth=0)
    axes[0].set_ylabel("Inflation (accuracy points)")
    axes[0].set_title("Inflation, mean over doses", loc="left")
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[1].set_ylabel("Min-K%++ AUROC at ×16\n(bars measured from chance)")
    axes[0].set_yticks(np.arange(-2, 15, 2))
    axes[1].axhline(0.5, color=INK2, lw=0.8)
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_title("Detection of 16-copy items", loc="left")
    for ax in axes:
        ax.set_xticks(x, [ABL_LAB[r] for r in runs])
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3)
    return _save(fig, out, "fig08_ablations")


# ------------------------------------------------------------------ error analysis
def fig_errors(counts, by_overlap, out):
    """Emphasis on misses (status red); everything else in neutral greys."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw={"width_ratios": [1, 1.15]})
    order = [o for o in ["hit", "false alarm", "miss", "neither"] if o in counts.index]
    shades = {"hit": INK2, "false alarm": REF_GREY, "miss": CRITICAL, "neither": LIGHT_GREY}
    bo = by_overlap.reindex(index=C.OVERLAPS, columns=order, fill_value=0)
    ax = axes[0]
    infl = bo[["hit", "miss"]]
    share_missed = infl["miss"] / infl.sum(1)
    yy = np.arange(len(C.OVERLAPS))
    ax.barh(yy, infl["hit"], color=INK2, height=0.58, label="inflated, flagged (hit)")
    ax.barh(yy, infl["miss"], left=infl["hit"], color=CRITICAL, height=0.58, label="inflated, not flagged (miss)")
    for i, o in enumerate(C.OVERLAPS):
        ax.text(infl.loc[o].sum() + 3, i, f"{share_missed[o]:.0%} missed", va="center", fontsize=7, color=INK)
    ax.set_yticks(yy, [OVERLAP_LAB[o] for o in C.OVERLAPS])
    ax.invert_yaxis()
    ax.set_xlim(0, 192)
    ax.set_xlabel("Inflated contaminated items")
    ax.set_title("Inflated items: caught vs missed", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, handlelength=1, handleheight=0.8)
    ax = axes[1]
    mt = [m for m in ["recitation", "reconstruction", "recollection"] if m in counts.columns]
    tot = counts.loc[order, mt].sum(1)
    mshade = {"recitation": INK, "reconstruction": INK2, "recollection": REF_GREY}
    left = np.zeros(len(order))
    for m in mt:
        v = counts.loc[order, m].to_numpy() / tot.to_numpy()
        ax.barh(range(len(order)), v, left=left, color=mshade[m], height=0.58, label=m, edgecolor=SURFACE, linewidth=1)
        left += v
    ax.set_yticks(range(len(order)), [f"{o} (n={int(tot[o])})" for o in order])
    for t in ax.get_yticklabels():
        if t.get_text().startswith("miss"):
            t.set_color(CRITICAL); t.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share by memorization type")
    ax.set_title("What each outcome consists of", loc="left")
    ax.legend(loc="outside lower center", ncol=3, bbox_to_anchor=None) if False else ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, handlelength=1, handleheight=0.8)
    for a_ in axes:
        a_.grid(axis="y", visible=False)
        a_.tick_params(axis="y", length=0)
    return _save(fig, out, "fig09_error_analysis")


# ------------------------------------------------------------------ comparison with literature
def fig_literature(det, inter, out):
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    ref = [("Fu et al.: pretraining", 0.49, 0.53), ("Min-K%++ paper: MIMIR", 0.52, 0.59),
           ("FSD paper: Pile", 0.57, 0.63), ("Fu et al.: fine-tuning", 0.41, 0.994)]
    for i, (lab, lo, hi) in enumerate(ref):
        ax.plot([lo, hi], [i, i], color=REF_GREY, lw=4.5, solid_capstyle="butt")
    ours = det.groupby("overlap")["auroc"].agg(["min", "max"]).reindex(C.OVERLAPS)
    for k, o in enumerate(C.OVERLAPS):
        i = len(ref) + k + 0.4
        ax.plot([ours.loc[o, "min"], ours.loc[o, "max"]], [i, i], color=OVERLAP_COL[o], lw=4.5, solid_capstyle="butt")
    ax.set_yticks([*range(len(ref)), *[len(ref) + k + 0.4 for k in range(3)]],
                  [r[0] for r in ref] + [f"This study: {OVERLAP_LAB[o].lower()}" for o in C.OVERLAPS])
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.5, -0.75, "chance", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xlabel("Reported AUROC range")
    ax.set_xlim(0.38, 1.02)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    p1 = _save(fig, out, "fig10_literature_comparison")

    fig, ax = plt.subplots(figsize=(COL, 2.0))
    M = inter.to_numpy().astype(float)
    n = len(inter)
    mask = np.triu(np.ones_like(M, dtype=bool))
    shown = np.ma.array(M, mask=mask)
    greys = LinearSegmentedColormap.from_list("g", ["#f0efec", "#52514e"])
    im = ax.imshow(shown, cmap=greys, vmin=0, vmax=1)
    ax.set_xticks(range(n - 1), inter.columns[:-1], rotation=30, ha="right")
    ax.set_yticks(range(1, n), inter.index[1:])
    ax.set_xlim(-0.5, n - 1.5); ax.set_ylim(n - 0.5, 0.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for (i, j), v in np.ndenumerate(M):
        if i > j:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK)
    _cbar(fig, im, ax, "Spearman ρ between detectors")
    _save(fig, out, "fig14_detector_agreement")
    return p1


def fig_control(ctrl, out):
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    lab = {"seed1": "Seed 1", "seed2": "Seed 2", "seed3": "Seed 3", "control": "Format-only control"}
    d = ctrl.set_index("run")
    names = [r for r in lab if r in d.index]
    v = d.loc[names, "mean_delta_acc_pts_clean_items"]
    ax.barh([lab[n] for n in names], v, color=[INK2 if n == "control" else REF_GREY for n in names], height=0.58)
    for i, x in enumerate(v):
        ax.text(x + 0.2, i, f"{x:+.1f}", va="center", fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 17.5)
    ax.grid(False)
    ax.set_title("Same gain with no inserted items", loc="left", fontsize=7.5)
    ax.set_xlabel("Accuracy gain on never-inserted items (points)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return _save(fig, out, "fig11_format_control")


def fig_training_loss(root, out):
    """All runs in grey (the point is stability); the two shorter runs are labelled directly."""
    files = sorted(Path(root, "results").glob("*_trainloss.csv"))
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    for f in files:
        d = pd.read_csv(f)
        name = f.stem.replace("_trainloss", "")
        y = d["loss"].rolling(10, min_periods=1).mean()
        short = name in ("pilot", "dilution")
        ax.plot(d["step"], y, lw=0.9 if not short else 1.2,
                color=REF_GREY if not short else {"pilot": INK, "dilution": "#8a3b12"}[name],
                alpha=0.9, zorder=2 if short else 1)
        if short:
            ax.annotate({"pilot": "pilot (1M filler)", "dilution": "half filler"}[name], (d["step"].iloc[-1], y.iloc[-1]),
                        xytext=(4, -10), textcoords="offset points", fontsize=6.5,
                        color={"pilot": INK, "dilution": "#8a3b12"}[name],
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))
    ax.plot([], [], color=REF_GREY, lw=0.9, label="seeds 1–3, control, question-only, rank 8")
    ax.legend(loc="upper right", fontsize=6.5)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel("Training loss (10-step mean)")
    return _save(fig, out, "fig12_training_loss")


def make_all(res: dict, root: Path = C.ROOT, out: Path | None = None) -> list[Path]:
    out = Path(out or Path(root) / "figures")
    ext = res.get("predictor_extended")
    if ext is None and Path(root, "tables", "supplement_predictor_extended.csv").exists():
        ext = pd.read_csv(Path(root, "tables", "supplement_predictor_extended.csv"))
    paths = [fig_inflation_heatmap(res["cell"], out), fig_inflation_dose(res["cell"], out),
             fig_auroc_heatmap(res["detection"], out), fig_roc_curves(res["avg"], out),
             fig_scatter(res["avg"], out, "det_minkpp"), fig_scatter(res["avg"], out, "det_ppl"),
             fig_forest(res["agreement"], out), fig_predictor(res["predictor"], out, ext),
             fig_ablation(res["ablation"], out), fig_errors(res["err_counts"], res["err_by_overlap"], out),
             fig_literature(res["detection"], res["inter_detector"], out), fig_control(res["control"], out),
             fig_training_loss(root, out), fig_cell_vs_item(res["cell"], res["detection"], res["agreement"], out)]
    return [p for p in paths if p]
