# Do contamination detectors predict score inflation?

**DAV research project, Phase 3 (Implementation and Experimental Evaluation)**
Muhammad Armughan Bakhshi (24I-2626) and Dawar Ahmed (24I-2533), BS DS 5B, FAST-NUCES Islamabad.

**Research question.** In small open LLMs, how well do contamination-detector scores predict item-level accuracy
inflation across exposure doses (x1, x4, x16) and overlap types (exact copy, near-duplicate, paraphrase)?

We fine-tune Pythia-410M (LoRA) on FineWeb-Edu text into which ARC-Easy test items are inserted at known doses and
overlap types. For every item we measure the score gain (inflation) against matched clean items, and the scores of
five detectors (Perplexity, Zlib, Min-K%, Min-K%++, FSD). We then test whether detector scores predict inflation.

## Repository layout

| Path | Contents |
|---|---|
| `dcl/config.py` | All settings: models, doses, cell sizes, seeds, run definitions |
| `dcl/data.py` | ARC-Easy loading, text template, n-grams, Pile check (infini-gram), matched assignment |
| `dcl/variants.py` | Near-duplicates and paraphrases (Qwen2.5-3B-Instruct + MiniLM similarity filter) |
| `dcl/mixture.py` | FineWeb-Edu filler (ARC-overlap filtered), contaminated copies, packing |
| `dcl/modeling.py` | Answer scoring, the 5 detectors, LoRA / full training loop, FSD |
| `dcl/pipeline.py` | `prepare_data()` (notebook 01) and `run_experiment()` (notebook 02) |
| `dcl/analysis.py` | Inflation, AUROC, agreement, combined predictor, mixed model, McNemar, DeLong, ablations, error analysis, literature comparison |
| `dcl/figures.py`, `dcl/dashboard.py` | 12 report figures and the interactive dashboard |
| `notebooks/` | `01_prepare_data`, `02_train_evaluate`, `03_analysis_figures` (Kaggle-ready) |
| `data/` | `items.csv` (items, groups, variants), `control_items.csv`, paraphrase manual check, logs |
| `results/` | One CSV per run (item-level scores), training loss and run logs |
| `tables/`, `figures/`, `dashboard/` | Analysis outputs |
| `tests/` | Offline smoke test (tiny random model) and synthetic-results test for notebook 03 |
| `commit_and_push.bat` | Double-click on Windows to commit and push this folder |

## How to run (Kaggle, free GPU)

One-time setup: verify your phone number on Kaggle (Settings > Phone verification); this unlocks GPU and internet.
In each notebook set **Accelerator = GPU T4 x1** and **Internet = On** (notebook 03 needs no GPU).
Import a notebook with *File > Import Notebook > GitHub* or upload the `.ipynb` file.

1. **Notebook 01 (Dawar, once, about 35 min).** Run all. Download `outputs_01.zip` from the Output panel, unzip it into
   this folder, double-click `commit_and_push.bat`. Fill in `data/paraphrase_manual_check.csv` (50 rows) and commit.
2. **Notebook 02, pilot (about 15 min).** `RUNS_TO_DO = ["pilot"]`. Check the printed inflation table: exact x16
   should be clearly positive. If it is close to zero, raise `lr` in `dcl/config.py` before the real runs.
3. **Notebook 02, core runs (about 30 min each), in parallel on two accounts:**
   Dawar `["seed1", "seed2", "seed3"]`, Armughan `["control", "qonly", "dilution", "rank8"]`.
   Optional if time allows: `["fullft160m", "rank64", "filler10m"]`.
   After each session download the `results_<run>.zip` files, unzip into this folder, run `commit_and_push.bat`.
4. **Notebook 03 (CPU, about 5 min).** Produces `tables/`, `figures/`, `dashboard/dashboard.html`. Commit them.

Run locally instead: `pip install -r requirements.txt`, then run the notebooks from this folder (a GPU is needed for 01-02).

## Experimental setup (reproducibility)

| Item | Value |
|---|---|
| Benchmark | ARC-Easy test split (4-option items), `allenai/ai2_arc` |
| Prior-exposure check | 13-word windows (start, middle, end) queried in the Pile via infini-gram (`v4_piletrain_llama`); found items removed |
| Split (seed 2026) | 9 cells x 150 items, 450 clean, 300 FSD reference, rest spare; matched on base-model difficulty quintile and question-length tercile |
| Overlap types | exact; near-duplicate (15% of words replaced/inserted/swapped); paraphrase (Qwen2.5-3B-Instruct, kept if MiniLM cosine >= 0.85 and no shared 8-gram; up to 4 tries) |
| Template | `Question: {question}\nAnswer: {answer}` (answer always kept, except the question-only ablation) |
| Filler | FineWeb-Edu `sample-10BT`, streamed in fixed order, documents with any ARC 13-gram removed; 5M tokens (main) |
| Model and training | Pythia-410M-deduped; LoRA r=16, alpha=32, dropout 0.05 on attention and MLP layers; 1 epoch; lr 2e-4, linear warm-up 3% then linear decay; 16 x 512-token sequences per step; fp16 autocast; AdamW |
| Seeds | 3 (seed controls data order and LoRA initialisation) |
| Answer scoring | length-normalised log-likelihood of each option; softmax over 4 options gives p(correct); accuracy = argmax |
| Detectors (on the original item text) | Perplexity (-loss); Zlib (-loss sum / zlib bytes); Min-K% (k=20%); Min-K%++ (k=20%); FSD on perplexity (LoRA r=8, 3 epochs, lr 1e-4 on the 300 reference items) |
| Reference method | 13-gram overlap between each test item and its inserted copy (needs training data; not counted as a detector) |
| Inflation | change in p(correct) (and accuracy) from the base model, minus the mean change on clean items |
| Statistics | bootstrap 95% CIs (10,000 for means, 1,000-2,000 for AUROC and rho); linear mixed model with item random intercept; McNemar (exact); DeLong; Holm correction |

## Phase 3 requirement map

| Handout item | Where |
|---|---|
| 1. Implementation | `dcl/`, notebooks 01-02 |
| 2. Dataset and preprocessing | notebook 01, `data/`, table above |
| 3. Baselines | 5 detectors + reference method + dose-only / difficulty-only / random predictors (`tables/detection.csv`, `predictor.csv`) |
| 4. Graphs | `figures/` (12 figures) and `dashboard/dashboard.html` |
| 5. Comparison with existing research | `tables/literature.csv`, Figure 10 |
| 6. Ablations | `qonly`, `dilution`, `rank8` runs (+ optional), detector settings, leave-one-detector-out (`tables/ablation.csv`, `detector_settings.csv`, `predictor.csv`) |
| 7. Discussion | written in the report from `tables/` |

## Testing without a GPU
`python tests/smoke_setup.py /tmp/smoke && python tests/smoke_run.py /tmp/smoke` runs the whole pipeline on a tiny random
model and fake data. `python tests/fake_results.py /tmp/smoke/work` writes synthetic results to test notebook 03.
Synthetic results are for code testing only.
