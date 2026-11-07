"""Training mixture: FineWeb-Edu filler + contaminated copies (Milestone 2, Step 3).

The filler is streamed in a fixed order, so every run and every team member gets the
same text without sharing large files. Documents containing any ARC 13-gram are removed.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from . import config as C
from .data import answer_text, item_text, ngrams, parse_group


def arc_ngram_set(frames: list[pd.DataFrame]) -> set[tuple]:
    s: set[tuple] = set()
    for df in frames:
        for _, r in df.iterrows():
            s |= ngrams(r["question"] + " " + answer_text(r), C.NGRAM_N)
    return {g for g in s if len(g) == C.NGRAM_N}


def stream_filler(tokenizer, n_tokens: int, banned: set[tuple], texts=None, log_every: int = 2000):
    """Returns (docs, stats). `texts` lets tests pass an iterable instead of FineWeb-Edu."""
    if texts is None:
        from datasets import load_dataset
        ds = load_dataset(C.FILLER_DATASET[0], name=C.FILLER_DATASET[1], split="train", streaming=True)
        texts = (ex["text"] for ex in ds)
    docs, total, dropped, seen = [], 0, 0, 0
    for t in texts:
        seen += 1
        if banned and (ngrams(t, C.NGRAM_N) & banned):
            dropped += 1
            continue
        n = len(tokenizer(t, add_special_tokens=False)["input_ids"])
        docs.append(t)
        total += n
        if log_every and len(docs) % log_every == 0:
            print(f"  filler: {len(docs):,} docs, {total/1e6:.2f}M tokens")
        if total >= n_tokens:
            break
    return docs, {"filler_docs": len(docs), "filler_tokens": total, "docs_seen": seen,
                  "docs_dropped_arc_overlap": dropped}


def contamination_docs(items: pd.DataFrame, run, control_items: pd.DataFrame | None = None) -> list[str]:
    """Copies of the contaminated test items (or ARC train items for the control run)."""
    docs = []
    if not run.contaminate:
        rnd = np.random.default_rng(C.SPLIT_SEED)
        doses = rnd.choice(C.DOSES, size=len(control_items))
        for (_, r), d in zip(control_items.iterrows(), doses):
            docs += [item_text(r["question"], answer_text(r))] * int(d)
        return docs
    for _, r in items.iterrows():
        dose, overlap = parse_group(r["group"])
        if dose == 0:
            continue
        q = {"exact": r["question"], "near_dup": r["near_dup"], "paraphrase": r["paraphrase"]}[overlap]
        a = answer_text(r) if run.answer_in_copy else None
        docs += [item_text(q, a)] * dose
    return docs


def training_text_of(row, run) -> str:
    """The text of the copy that was inserted for this item ('' if clean)."""
    dose, overlap = parse_group(row["group"])
    if dose == 0 or not run.contaminate:
        return ""
    q = {"exact": row["question"], "near_dup": row["near_dup"], "paraphrase": row["paraphrase"]}[overlap]
    return item_text(q, answer_text(row) if run.answer_in_copy else None)


def pack(tokenizer, docs: list[str], seq_len: int, seed: int):
    """Shuffle documents, join with EOS and cut into fixed-length blocks."""
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    enc = tokenizer([docs[i] for i in order], add_special_tokens=False)["input_ids"]
    for e in enc:
        ids.extend(e)
        ids.append(eos)
    n = len(ids) // seq_len
    arr = np.array(ids[: n * seq_len], dtype=np.int64).reshape(n, seq_len)
    return arr
