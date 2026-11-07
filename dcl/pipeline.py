"""High-level steps called by the notebooks.

prepare_data()  -> notebook 01 (Steps 1-4 of the methodology)
run_experiment() -> notebook 02 (Steps 5-7: train one run, evaluate, score detectors)
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from . import mixture as M
from . import modeling as MD
from . import variants as V


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================ notebook 01
def prepare_data(out_root: Path = C.ROOT, base_model: str = C.BASE_MODEL, arc_test=None, arc_train=None,
                 paraphraser=None, similarity=None, check_pile: bool = True):
    t0 = time.time()
    out_root = Path(out_root)
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    (out_root / "results").mkdir(parents=True, exist_ok=True)
    log = {}
    set_seed(C.SPLIT_SEED)

    print("1/6 Loading ARC-Easy ...")
    test = arc_test if arc_test is not None else D.load_arc("test")
    train = arc_train if arc_train is not None else D.load_arc("train")
    log["arc_test_4option"] = len(test)
    log["arc_train_4option"] = len(train)
    print(f"   test items (4 options): {len(test)}   train items: {len(train)}")

    print("2/6 Prior-exposure check against the Pile (infini-gram) ...")
    if check_pile:
        counts = D.pile_exposure(test)
        test["pile_count"] = counts
        found = (counts > 0).sum()
        unknown = counts.isna().sum()
        log.update(pile_found=int(found), pile_unknown=int(unknown))
        print(f"   found in Pile: {found}   API unavailable for: {unknown}")
        need = C.CELL_SIZE * 14
        keep = ~(counts > 0)
        if keep.sum() < need:   # not enough unseen items: keep the least-exposed found items and flag them
            extra = counts[counts > 0].sort_values().index[: need - int(keep.sum())]
            keep.loc[extra] = True
            print(f"   only {int((~(counts > 0)).sum())} unseen items; keeping {len(extra)} least-exposed found items (flagged)")
            log["pile_found_kept"] = int(len(extra))
        test["pile_flag"] = (counts > 0).astype(int)
        test = test[keep].reset_index(drop=True)
    else:
        test["pile_count"] = np.nan
        test["pile_flag"] = 0
        log.update(pile_found=None, pile_unknown=len(test))
        print("   skipped")
    log["items_after_pile_check"] = len(test)

    print("3/6 Scoring the base model (difficulty for matching + base detector scores) ...")
    model, tok = MD.load_model(base_model)
    base = MD.score_options(model, tok, test)
    det = MD.detector_scores(model, tok, MD.detector_texts(test))
    base = pd.concat([base.reset_index(drop=True), det], axis=1)
    test["base_p_correct"] = base["p_correct"].to_numpy()
    log["base_accuracy"] = float(base["acc"].mean())
    print(f"   base accuracy: {base['acc'].mean():.3f}   mean p_correct: {base['p_correct'].mean():.3f}")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("4/6 Difficulty-matched assignment to cells ...")
    test["group"] = D.assign_groups(test)
    print("   " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(test["group"]).items())))

    print("5/6 Near-duplicates ...")
    vocab = [w for w, _ in Counter(w for q in test["question"] for w in q.split()).most_common(300)]
    test["near_dup"] = [V.near_duplicate(q, seed=C.SPLIT_SEED + i, vocab=vocab)
                        for i, q in enumerate(test["question"])]

    print("6/6 Paraphrases (instruction model + similarity filter) ...")
    para_mask = test["group"].str.startswith("paraphrase")
    if paraphraser is None:
        paraphraser = V.QwenParaphraser()
    if similarity is None:
        similarity = V.MiniLMSimilarity()
    qs = test.loc[para_mask, "question"].tolist()
    best, sims, passed, attempts = V.make_paraphrases(qs, paraphraser, similarity)
    test["paraphrase"] = ""
    test["para_sim"] = np.nan
    test["para_passed"] = np.nan
    test.loc[para_mask, "paraphrase"] = best
    test.loc[para_mask, "para_sim"] = sims
    test.loc[para_mask, "para_passed"] = passed.astype(float)
    # items outside paraphrase cells get a placeholder so the column is never empty
    test.loc[~para_mask, "paraphrase"] = test.loc[~para_mask, "question"]
    log.update(paraphrase_items=int(para_mask.sum()), paraphrase_pass_rate=float(passed.mean()),
               paraphrase_mean_sim=float(np.mean(sims)), paraphrase_mean_attempts=float(attempts.mean()))
    print(f"   pass rate: {passed.mean():.1%}   mean similarity: {np.mean(sims):.3f}")

    # ---------------- save
    base.insert(0, "group", test["group"].to_numpy())
    test.to_csv(out_root / "data" / "items.csv", index=False)
    train.sample(n=min(C.N_CONTROL_TRAIN, len(train)), random_state=C.SPLIT_SEED) \
         .to_csv(out_root / "data" / "control_items.csv", index=False)
    base.to_csv(out_root / "results" / "base.csv", index=False)
    chk = test[para_mask].sample(n=min(50, int(para_mask.sum())), random_state=C.SPLIT_SEED)
    chk = chk[["item_id", "question", "paraphrase", "para_sim", "para_passed"]].copy()
    chk["meaning_kept (Y/N)"] = ""
    chk.to_csv(out_root / "data" / "paraphrase_manual_check.csv", index=False)
    log["minutes"] = round((time.time() - t0) / 60, 1)
    (out_root / "data" / "prepare_log.json").write_text(json.dumps(log, indent=2))
    print(f"Done in {log['minutes']} min. Files written to data/ and results/base.csv")
    return test, base, log


# ============================================================ notebook 02
def run_experiment(run_name: str, root: Path = C.ROOT, out_root: Path | None = None, filler_texts=None,
                   **overrides):
    t0 = time.time()
    run = C.get_run(run_name, **overrides)
    root = Path(root)
    out_root = Path(out_root or root)
    (out_root / "results").mkdir(parents=True, exist_ok=True)
    set_seed(run.seed)
    items = pd.read_csv(root / "data" / "items.csv")
    control = pd.read_csv(root / "data" / "control_items.csv")
    print(f"Run '{run.name}': model={run.model}  seed={run.seed}  filler={run.filler_tokens/1e6:.1f}M  "
          f"lora={'r'+str(run.lora_r) if run.lora else 'off (full fine-tuning)'}  lr={run.lr}")
    log = {"run": run.__dict__.copy()}

    model, tok = MD.load_model(run.model)
    own_base = None
    if run.model != C.BASE_MODEL:
        print("Scoring this run's own base model ...")
        own_base = MD.score_options(model, tok, items)

    print("Building the training mixture ...")
    banned = M.arc_ngram_set([items, control])
    filler, fstats = M.stream_filler(tok, run.filler_tokens, banned, texts=filler_texts)
    cdocs = M.contamination_docs(items, run, control_items=control)
    blocks = M.pack(tok, filler + cdocs, C.SEQ_LEN, seed=run.seed)
    ctok = sum(len(e) + 1 for e in tok(cdocs, add_special_tokens=False)["input_ids"]) if cdocs else 0
    mstats = {**fstats, "copy_docs": len(cdocs), "copy_tokens": ctok,
              "copy_share": ctok / max(1, ctok + fstats["filler_tokens"]), "train_blocks": int(len(blocks))}
    log["mixture"] = mstats
    print(f"   {len(cdocs):,} inserted copies = {mstats['copy_share']:.1%} of training tokens; "
          f"{len(blocks):,} blocks of {C.SEQ_LEN}")

    print("Training ...")
    model = MD.make_trainable(model, run)
    tl = MD.train_blocks(model, blocks, lr=run.lr, batch_size=run.batch_size, micro_batch=run.micro_batch,
                         epochs=run.epochs, seed=run.seed)
    log["final_train_loss"] = float(tl["loss"].tail(10).mean())

    print("Evaluating answers and detectors on all items ...")
    res = MD.score_options(model, tok, items)
    texts = MD.detector_texts(items)
    det = MD.detector_scores(model, tok, texts)
    res = pd.concat([res.reset_index(drop=True), det], axis=1)
    res.insert(1, "group", items["group"].to_numpy())
    res["ngram_ref"] = [D.overlap_fraction(t, M.training_text_of(r, run))
                        for t, (_, r) in zip(texts, items.iterrows())]
    if own_base is not None:
        res["own_base_p_correct"] = own_base["p_correct"].to_numpy()
        res["own_base_acc"] = own_base["acc"].to_numpy()

    print("FSD: short fine-tuning on the reference pool, then re-scoring ...")
    ref = items[items["group"] == "fsd_ref"]
    fsd = MD.fsd_scores(model, tok, run, texts, MD.detector_texts(ref), det)
    res = pd.concat([res, fsd], axis=1)
    res.insert(0, "run", run.name)

    res.to_csv(out_root / "results" / f"{run.name}.csv", index=False)
    tl.to_csv(out_root / "results" / f"{run.name}_trainloss.csv", index=False)
    log["minutes"] = round((time.time() - t0) / 60, 1)
    log["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    (out_root / "results" / f"{run.name}_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"Done in {log['minutes']} min -> results/{run.name}.csv")
    quick_summary(res, root)
    return res, log


def quick_summary(res: pd.DataFrame, root: Path = C.ROOT):
    """Prints mean change in p_correct per cell vs the clean items (sanity check)."""
    base_path = Path(root) / "results" / "base.csv"
    if "own_base_p_correct" in res:
        bp = res["own_base_p_correct"].to_numpy()
    elif base_path.exists():
        bp = pd.read_csv(base_path).set_index("item_id").loc[res["item_id"], "p_correct"].to_numpy()
    else:
        return
    d = res.assign(delta=res["p_correct"].to_numpy() - bp)
    clean = d.loc[d["group"] == "clean", "delta"].mean()
    tab = d.groupby("group")["delta"].mean().sub(clean).drop(["fsd_ref", "spare"], errors="ignore")
    print("\nInflation check (mean change in p_correct minus clean items' change):")
    for g, v in tab.items():
        print(f"   {g:16s} {v:+.3f}")
