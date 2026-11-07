"""Generates the three Kaggle notebooks in notebooks/ (run: python tests/make_notebooks.py)."""
from pathlib import Path

import nbformat as nbf

REPO_URL = "https://github.com/DawarAhmed27/DAV-Reseach-DataContamination.git"
OUT = Path(__file__).resolve().parent.parent / "notebooks"

SETUP = f'''# --- Get the project code from GitHub (works for public repos; for a private repo add a Kaggle
#     secret called GITHUB_TOKEN: Add-ons > Secrets) ---
import os, subprocess, sys
from pathlib import Path
REPO_URL = "{REPO_URL}"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
REPO = WORK / "repo"
url = REPO_URL
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
    url = REPO_URL.replace("https://", f"https://{{tok}}@")
except Exception:
    pass
if REPO.exists():   # update to the latest GitHub version (local generated files are replaced by GitHub's copy)
    subprocess.run(["git", "-C", str(REPO), "fetch", "--quiet"], check=False)
    subprocess.run(["git", "-C", str(REPO), "reset", "--quiet", "--hard", "@{{u}}"], check=False)
else:
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", url, str(REPO)], check=True)
sys.path.insert(0, str(REPO))
print("Code ready in", REPO)'''

INSTALL = '''# --- Libraries that Kaggle does not always have ---
!pip uninstall -y -q torchao 2>/dev/null   # Kaggle's old torchao breaks the latest peft; we do not use it
!pip install -q peft sentence-transformers 2>/dev/null | tail -1
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - turn on GPU T4 in Settings > Accelerator")'''


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def save(cells, name):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}}
    OUT.mkdir(exist_ok=True)
    nbf.write(nb, OUT / name)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 01
save([
    md("""# 01 · Prepare the data  (run ONCE, by Dawar)
**Kaggle settings (right panel):** Accelerator = **GPU T4 x1** (or P100), Internet = **On**.  Run time: about 30-40 min.

What this notebook does (Milestone 2, Steps 1-4):
1. Loads the ARC-Easy test set (4-option items) and 1,000 training items for the format-only control.
2. Checks every test item against the Pile with the infini-gram API and drops items Pythia has already seen.
3. Scores the base Pythia-410M model on every item (difficulty and base detector scores).
4. Assigns items to the 9 contaminated cells (150 each), 450 clean items and 300 FSD reference items, matched on difficulty and length.
5. Builds near-duplicates and generates paraphrases with Qwen2.5-3B-Instruct (filtered by similarity >= 0.85 and no shared 8-word sequence).

**Output:** `outputs_01.zip` (in the Output panel on the right). Download it, unzip it **into your local repo folder** (it contains `data/` and `results/`), then double-click `commit_and_push.bat`. Notebook 02 reads these files from GitHub."""),
    code(SETUP),
    code(INSTALL),
    code('''from dcl import pipeline as P
items, base, log = P.prepare_data(out_root=REPO)
log'''),
    code('''# Quick checks: group sizes and difficulty balance (should be similar across groups)
import pandas as pd
print(items["group"].value_counts().sort_index().to_string())
print(items.groupby(items["group"].str.replace(r"_x\\d+", "", regex=True))["base_p_correct"].mean().round(3).to_string())'''),
    code('''# Package the outputs for download
import shutil, zipfile
zpath = WORK / "outputs_01.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p in list((REPO / "data").glob("*")) + [REPO / "results" / "base.csv"]:
        if p.is_file() and p.name != ".gitkeep":
            z.write(p, p.relative_to(REPO))
print("Download", zpath, "from the Output panel, unzip into your repo folder, then run commit_and_push.bat")'''),
    md("""**Before notebook 02:** open `data/paraphrase_manual_check.csv`, fill the column `meaning_kept (Y/N)` for the 50 rows (Armughan can do this), and commit it. The result goes in the report."""),
], "01_prepare_data.ipynb")

# ---------------------------------------------------------------- 02
save([
    md("""# 02 · Train and evaluate one or more runs
**Kaggle settings:** Accelerator = **GPU T4 x1**, Internet = **On**. About 25-35 min per run.

| Who | Runs |
|---|---|
| First, once | `pilot` (about 15 min) - send the printed inflation check before the real runs |
| Dawar | `seed1`, `seed2`, `seed3` |
| Armughan | `control`, `qonly`, `dilution`, `rank8` |
| Optional, if time | `fullft160m`, `rank64`, `filler10m` |

Each run: builds the training mix (FineWeb-Edu + contaminated copies), fine-tunes Pythia-410M with LoRA, scores every item,
runs the 5 detectors (Perplexity, Zlib, Min-K%, Min-K%++, FSD) and the 13-gram reference check.

**Output:** `results_<run>.zip` per run. Download, unzip into the repo folder, double-click `commit_and_push.bat`.
Notebook 01's files must already be on GitHub."""),
    code('''# ===== CHOOSE THE RUNS FOR THIS SESSION =====
RUNS_TO_DO = ["pilot"]          # e.g. ["seed1", "seed2", "seed3"]  or  ["control", "qonly", "dilution", "rank8"]'''),
    code(SETUP),
    code(INSTALL),
    code('''assert (REPO / "data" / "items.csv").exists(), "data/items.csv not found - run notebook 01 and push its outputs first"
from dcl import pipeline as P
import zipfile, traceback
for run in RUNS_TO_DO:
    print("=" * 70)
    try:
        P.run_experiment(run, root=REPO)
        z = WORK / f"results_{run}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in (REPO / "results").glob(f"{run}*"):
                zf.write(p, p.relative_to(REPO))
        print("Saved", z)
    except Exception:
        traceback.print_exc()
        print(f"Run {run} failed - copy the error above and send it.")'''),
], "02_train_evaluate.ipynb")

# ---------------------------------------------------------------- 03
save([
    md("""# 03 · Analysis, figures and dashboard  (CPU is enough)
Run after all result files are on GitHub (at least `seed1`-`seed3`; ablations are used if present).
Works on Kaggle (Accelerator = None, Internet = On) or on your laptop (`pip install -r requirements.txt`, then run from the repo folder). About 5 min.

**Output:** `outputs_03.zip` with `tables/` (all CSV tables), `figures/` (12 PNG figures) and `dashboard/dashboard.html`.
Unzip into the repo folder and commit; then send the zip so the report can be written."""),
    code(SETUP.replace('if REPO.exists():', 'if Path("dcl").exists():\n    REPO = Path.cwd()          # running locally inside the repo folder\nelif REPO.exists():')),
    code('''!pip install -q plotly statsmodels 2>/dev/null | tail -1
from dcl import analysis as A, figures as F, dashboard as DB
res = A.run_all(REPO)
print("Runs found:", sorted(res["runs"]))
res["cell"][["group", "n_items", "infl_p", "infl_p_lo", "infl_p_hi", "infl_acc_pts"]].round(3)'''),
    md("## RQ2 - detection (AUROC per cell)"),
    code('res["detection"].pivot_table(index="detector", columns="group", values="auroc").round(3)'),
    md("## RQ3 - does the detector score predict inflation?"),
    code('''display(res["agreement"][res["agreement"]["scope"] != "within cell"].round(3))
display(res["agreement"].groupby("detector")["mean_within_cell_rho"].first().round(3))
display(res["partial"].round(3))
res["predictor"].round(3)'''),
    md("## Statistics, ablations, control, literature comparison"),
    code('''display(res["mixed"].round(4))
display(res["mcnemar"].round(4))
display(res["ablation"].round(3).T)
display(res["detector_settings"].round(3))
display(res["control"].round(3))
res["literature"]'''),
    code('''paths = F.make_all(res, REPO)
dash = DB.build(res, REPO / "dashboard" / "dashboard.html")
import zipfile
z = WORK / "outputs_03.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in ["tables", "figures", "dashboard"]:
        for p in (REPO / folder).glob("*"):
            if p.is_file() and p.name != ".gitkeep":
                zf.write(p, p.relative_to(REPO))
print(len(paths), "figures;", dash, "; download", z)'''),
    code('''from IPython.display import Image, display
for p in paths:
    display(Image(filename=str(p), width=900))'''),
], "03_analysis_figures.ipynb")
