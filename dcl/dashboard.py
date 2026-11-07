"""Interactive dashboard (Milestone 2, Step 12): one self-contained HTML file, light theme."""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

from . import config as C
from .analysis import DETECTORS
from .figures import CAT, DET_COL, OVERLAP_COL, OVERLAP_LAB

FONT = dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color="#0b0b0b")
LAYOUT = dict(template="plotly_white", paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", font=FONT,
              margin=dict(l=60, r=20, t=50, b=50), hoverlabel=dict(bgcolor="white", font_size=12))
BLUE_SCALE = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]


def _heat_inflation(cell):
    c = cell[cell["dose"] > 0]
    z = c.pivot_table(index="overlap", columns="dose", values="infl_p").reindex(index=C.OVERLAPS, columns=C.DOSES)
    acc = c.pivot_table(index="overlap", columns="dose", values="infl_acc_pts").reindex(index=C.OVERLAPS, columns=C.DOSES)
    txt = [[f"{z.iloc[i, j]:+.3f}<br>({acc.iloc[i, j]:+.1f} pts)" for j in range(3)] for i in range(3)]
    fig = go.Figure(go.Heatmap(z=z.to_numpy(), x=[f"x{d}" for d in C.DOSES], y=[OVERLAP_LAB[o] for o in C.OVERLAPS],
                               colorscale=BLUE_SCALE, text=txt, texttemplate="%{text}", xgap=2, ygap=2,
                               hovertemplate="%{y}, %{x}<br>inflation %{z:+.3f}<extra></extra>",
                               colorbar=dict(title="p(correct)")))
    fig.update_layout(**LAYOUT, title="Score inflation by dose and overlap type", height=330)
    fig.update_yaxes(autorange="reversed")
    return fig


def _lines(cell):
    fig = go.Figure()
    for o in C.OVERLAPS:
        d = cell[cell["overlap"] == o].sort_values("dose")
        x = [f"x{v}" for v in d["dose"]]
        fig.add_trace(go.Scatter(x=x + x[::-1], y=list(d["infl_p_hi"]) + list(d["infl_p_lo"])[::-1], fill="toself",
                                 fillcolor=OVERLAP_COL[o], opacity=0.15, line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=d["infl_p"], name=OVERLAP_LAB[o], mode="lines+markers",
                                 line=dict(color=OVERLAP_COL[o], width=2), marker=dict(size=9),
                                 hovertemplate=OVERLAP_LAB[o] + " %{x}: %{y:+.3f}<extra></extra>"))
    fig.update_layout(**LAYOUT, title="Inflation vs dose (95% CI bands)", height=330, hovermode="x unified",
                      yaxis_title="inflation in p(correct)", xaxis_title="copies seen")
    return fig


def _auroc_heat(det):
    piv = det.pivot_table(index="detector", columns="group", values="auroc").reindex(index=list(DETECTORS.values()), columns=C.CELLS)
    fig = go.Figure(go.Heatmap(z=piv.to_numpy(), x=[g.replace("_x", " x").replace("near_dup", "near-dup") for g in C.CELLS],
                               y=piv.index, zmin=0.3, zmax=1.0, colorscale=[[0, "#eb6834"], [0.2857, "#f3f2ef"], [1, "#2a78d6"]],
                               texttemplate="%{z:.2f}", xgap=2, ygap=2, colorbar=dict(title="AUROC"),
                               hovertemplate="%{y} on %{x}: AUROC %{z:.3f}<extra></extra>"))
    fig.update_layout(**LAYOUT, title="Detection AUROC (cell vs clean items; 0.5 = chance)", height=330)
    fig.update_yaxes(autorange="reversed")
    return fig


def _roc(avg):
    clean = avg[avg["group"] == "clean"]
    fig = go.Figure()
    n_det = len(DETECTORS)
    for gi, g in enumerate(C.CELLS):
        d = avg[avg["group"] == g]
        y = np.r_[np.ones(len(d)), np.zeros(len(clean))]
        for col, lab in DETECTORS.items():
            f, t, _ = roc_curve(y, np.r_[d[col], clean[col]])
            fig.add_trace(go.Scatter(x=f, y=t, name=lab, line=dict(color=DET_COL[lab], width=2), visible=(g == "exact_x16"),
                                     hovertemplate=lab + ": FPR %{x:.2f}, TPR %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#a3a29b", dash="dash", width=1),
                             showlegend=False, hoverinfo="skip"))
    buttons = []
    for gi, g in enumerate(C.CELLS):
        vis = [False] * (len(C.CELLS) * n_det) + [True]
        vis[gi * n_det:(gi + 1) * n_det] = [True] * n_det
        buttons.append(dict(label=g.replace("_x", " x").replace("near_dup", "near-dup"), method="update",
                            args=[{"visible": vis}, {"title": f"ROC curves: {g.replace('_x', ' x')} vs clean"}]))
    fig.update_layout(**LAYOUT, title="ROC curves: exact x16 vs clean", height=420,
                      xaxis_title="false positive rate", yaxis_title="true positive rate",
                      updatemenus=[dict(buttons=buttons, active=C.CELLS.index("exact_x16"), x=1.0, xanchor="right", y=1.18,
                                        bgcolor="white", bordercolor="#d4d3cd")])
    return fig


def _scatter(avg, items):
    q = items.set_index("item_id")["question"]
    fig = go.Figure()
    dets = list(DETECTORS.items())
    for k, (col, lab) in enumerate(dets):
        for o in C.OVERLAPS:
            d = avg[avg["overlap"] == o]
            hover = [f"{html.escape(str(q.get(i, ''))[:90])}<br>{g} | score {s:.3f} | inflation {v:+.3f}"
                     for i, g, s, v in zip(d["item_id"], d["group"], d[col], d["infl_p"])]
            fig.add_trace(go.Scattergl(x=d[col], y=d["infl_p"], mode="markers", name=OVERLAP_LAB[o],
                                       marker=dict(color=OVERLAP_COL[o], size=7, opacity=0.7,
                                                   line=dict(color="#fcfcfb", width=0.5)),
                                       text=hover, hovertemplate="%{text}<extra></extra>", visible=(k == 3),
                                       legendgroup=o, showlegend=True))
        cl = avg[avg["group"] == "clean"]
        fig.add_trace(go.Scattergl(x=cl[col], y=cl["infl_p"], mode="markers", name="clean",
                                   marker=dict(color="#bdbcb6", size=5, opacity=0.6), visible=(k == 3),
                                   hovertemplate="clean item<extra></extra>"))
    per = len(C.OVERLAPS) + 1
    buttons = []
    for k, (col, lab) in enumerate(dets):
        vis = [False] * (per * len(dets))
        vis[k * per:(k + 1) * per] = [True] * per
        buttons.append(dict(label=lab, method="update", args=[{"visible": vis}, {"xaxis.title.text": f"{lab} score (higher = 'seen')"}]))
    fig.update_layout(**LAYOUT, title="Item-level detector score vs inflation (pick a detector)", height=460,
                      xaxis_title="Min-K%++ score (higher = 'seen')", yaxis_title="item inflation in p(correct)",
                      updatemenus=[dict(buttons=buttons, active=3, x=1.0, xanchor="right", y=1.15, bgcolor="white",
                                        bordercolor="#d4d3cd")])
    return fig


def _rho(agr):
    w = agr[agr["scope"] == "within cell"]
    fig = go.Figure()
    for lab in DETECTORS.values():
        d = w[w["detector"] == lab].set_index("group").reindex(C.CELLS)
        fig.add_trace(go.Bar(x=[g.replace("_x", " x").replace("near_dup", "near-dup") for g in C.CELLS], y=d["rho"],
                             name=lab, marker_color=DET_COL[lab],
                             error_y=dict(type="data", symmetric=False, array=d["rho_hi"] - d["rho"],
                                          arrayminus=d["rho"] - d["rho_lo"], color="#52514e", thickness=1),
                             hovertemplate=lab + " %{x}: ρ = %{y:.2f}<extra></extra>"))
    fig.update_layout(**LAYOUT, title="Within-cell Spearman ρ between detector score and inflation", barmode="group",
                      height=380, yaxis_title="Spearman ρ", bargap=0.25)
    return fig


def _ablation(abl):
    lab = {"seed1": "Main (seed 1)", "qonly": "Question only", "dilution": "Half filler", "rank8": "LoRA rank 8",
           "fullft160m": "Full FT 160M", "rank64": "LoRA rank 64", "filler10m": "Double filler"}
    a = abl[abl["run"].isin(lab)].set_index("run")
    fig = go.Figure()
    for o in C.OVERLAPS:
        fig.add_trace(go.Bar(x=[lab[r] for r in a.index], y=a[f"infl_p_{o}"], name=OVERLAP_LAB[o], marker_color=OVERLAP_COL[o],
                             hovertemplate=OVERLAP_LAB[o] + " %{x}: %{y:+.3f}<extra></extra>"))
    fig.update_layout(**LAYOUT, title="Ablations: mean inflation by overlap type", barmode="group", height=360,
                      yaxis_title="inflation in p(correct)")
    return fig


def _table(df, title, cols, fmt=None, height=None):
    fmt = fmt or {}
    vals = [[fmt.get(c, "{}").format(v) if isinstance(v, (int, float, np.floating)) and c in fmt else v for v in df[c]] for c in cols]
    fig = go.Figure(go.Table(header=dict(values=[f"<b>{c}</b>" for c in cols], fill_color="#eef3fb", align="left",
                                         font=dict(size=12)),
                             cells=dict(values=vals, fill_color="#fcfcfb", align="left", height=26, font=dict(size=11))))
    fig.update_layout(**{**LAYOUT, "margin": dict(l=10, r=10, t=45, b=10)}, title=title, height=height or 60 + 28 * (len(df) + 1))
    return fig


def build(res: dict, out_file: Path):
    avg, cell, det = res["avg"], res["cell"], res["detection"]
    ex16 = cell.set_index("group")
    best = det.groupby("group")["auroc"].max()
    w = res["agreement"]
    wmean = w[w["scope"] == "within cell"].groupby("detector")["rho"].mean()
    tiles = [("Inflation, exact x16", f"{ex16.loc['exact_x16', 'infl_acc_pts']:+.1f} pts"),
             ("Inflation, paraphrase x16", f"{ex16.loc['paraphrase_x16', 'infl_acc_pts']:+.1f} pts"),
             ("Best AUROC, exact x16", f"{best['exact_x16']:.2f}"),
             ("Best AUROC, paraphrase x16", f"{best['paraphrase_x16']:.2f}"),
             ("Mean within-cell ρ (best detector)", f"{wmean.max():.2f}")]
    figs = [_heat_inflation(cell), _lines(cell), _auroc_heat(det), _roc(avg), _scatter(avg, res["items"]),
            _rho(res["agreement"]), _ablation(res["ablation"]),
            _table(res["literature"], "Comparison with published results",
                   ["paper", "quantity", "reported", "ours", "limits_on_fair_comparison"], height=420),
            _table(res["err_samples"].head(60), "Error analysis: sample items (hits, false alarms, misses)",
                   ["outcome", "group", "mem_type", "infl_p", "det_minkpp", "question"],
                   fmt={"infl_p": "{:+.3f}", "det_minkpp": "{:.3f}"}, height=520)]
    parts = []
    for i, f in enumerate(figs):
        parts.append(f'<div class="card">{f.to_html(full_html=False, include_plotlyjs="inline" if i == 0 else False, config={"displaylogo": False})}</div>')
    tile_html = "".join(f'<div class="tile"><div class="tl">{html.escape(a)}</div><div class="tv">{html.escape(b)}</div></div>' for a, b in tiles)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Contamination Detector Dashboard</title>
<style>
:root {{ --surface:#fcfcfb; --page:#f3f2ef; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; color-scheme: light; }}
body {{ margin:0; background:var(--page); color:var(--ink); font-family:Inter,"Segoe UI",Helvetica,Arial,sans-serif; }}
header {{ padding:24px 24px 8px; max-width:1200px; margin:auto; }}
h1 {{ font-size:22px; margin:0 0 6px; }} p.sub {{ color:var(--ink2); margin:0 0 12px; font-size:14px; line-height:1.45; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; max-width:1200px; margin:0 auto; padding:0 24px; }}
.tile {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.tl {{ font-size:12px; color:var(--ink2); }} .tv {{ font-size:24px; font-weight:650; margin-top:4px; }}
main {{ max-width:1200px; margin:auto; padding:12px 24px 40px; display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:6px; min-width:0; }}
.card:nth-child(n+3) {{ grid-column:1 / -1; }}
@media (max-width:820px) {{ main {{ grid-template-columns:1fr; padding:12px 16px 32px; }} header, .tiles {{ padding-left:16px; padding-right:16px; }} }}
</style></head><body>
<header><h1>Do contamination detectors predict score inflation?</h1>
<p class="sub">Pythia-410M fine-tuned with LoRA on FineWeb-Edu text containing ARC-Easy test items at doses x1/x4/x16 as exact copies,
near-duplicates or paraphrases. Inflation = change in an item's p(correct) minus the average change on clean items.
Values are means of {avg.attrs.get('n_seeds', 3)} seeds. Hover any mark for details; use the menus to switch detector or cell.</p></header>
<section class="tiles">{tile_html}</section>
<main>{''.join(parts)}</main></body></html>"""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(page, encoding="utf-8")
    return out_file
