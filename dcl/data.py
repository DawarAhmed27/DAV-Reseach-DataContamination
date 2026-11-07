"""Loading ARC-Easy, text templates, n-gram utilities, prior-exposure check and
difficulty-matched assignment of items to cells (Milestone 2, Steps 2-3)."""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import config as C

LETTERS = "ABCDE"


# ------------------------------------------------------------------ loading
def load_arc(split: str = "test") -> pd.DataFrame:
    """ARC-Easy as a flat table; keeps only 4-option items."""
    from datasets import load_dataset
    ds = load_dataset(*C.ARC_DATASET, split=split)
    rows = []
    for ex in ds:
        texts, labels = ex["choices"]["text"], ex["choices"]["label"]
        if len(texts) != 4:
            continue
        key = ex["answerKey"]
        if key not in labels:
            continue
        rows.append({
            "item_id": ex["id"], "question": ex["question"].strip(),
            "choices": json.dumps([t.strip() for t in texts]),
            "answer_idx": labels.index(key),
        })
    return pd.DataFrame(rows)


def choices_of(row) -> list[str]:
    return json.loads(row["choices"]) if isinstance(row["choices"], str) else list(row["choices"])


def answer_text(row) -> str:
    return choices_of(row)[int(row["answer_idx"])]


# ------------------------------------------------------------------ templates
def prompt_text(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def item_text(question: str, answer: str | None) -> str:
    """Fixed template used both for training copies and for detector scoring."""
    if answer is None:
        return f"Question: {question}"
    return f"Question: {question}\nAnswer: {answer}"


# ------------------------------------------------------------------ n-grams
_word = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _word.findall(text.lower())


def ngrams(text: str, n: int) -> set[tuple]:
    w = words(text)
    if len(w) < n:
        return {tuple(w)} if w else set()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def shared_ngram(a: str, b: str, n: int) -> bool:
    wa, wb = words(a), words(b)
    if len(wa) < n or len(wb) < n:
        return False
    return bool(ngrams(a, n) & ngrams(b, n))


def overlap_fraction(test: str, train: str, n: int = C.NGRAM_N) -> float:
    """Share of the test text's n-grams (n shrinks for short texts) found in train text."""
    nn = min(n, max(1, len(words(test))))
    t = ngrams(test, nn)
    if not t:
        return 0.0
    return len(t & ngrams(train, nn)) / len(t)


# ------------------------------------------------------------------ prior exposure (infini-gram)
def _infini_count(query: str, session, retries: int = 3) -> int | None:
    payload = {"index": C.INFINIGRAM_INDEX, "query_type": "count", "query": query}
    for i in range(retries):
        try:
            r = session.post(C.INFINIGRAM_URL, json=payload, timeout=30)
            j = r.json()
            if "count" in j:
                return int(j["count"])
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def pile_exposure(df: pd.DataFrame, max_queries_per_item: int = 3, workers: int = 8) -> pd.Series:
    """Counts how often 13-word windows of each item (question + answer) appear in the Pile.

    Returns a Series: 0 = not found, >0 = found, NaN = API unavailable.
    Up to 3 windows per item (start, middle, end) keep the number of API calls small.
    """
    import requests
    session = requests.Session()

    def check(row):
        w = (row["question"] + " " + answer_text(row)).split()
        n = C.NGRAM_N
        if len(w) <= n:
            windows = [" ".join(w)]
        else:
            starts = sorted({0, (len(w) - n) // 2, len(w) - n})[:max_queries_per_item]
            windows = [" ".join(w[s:s + n]) for s in starts]
        counts = [_infini_count(q, session) for q in windows]
        if all(c is None for c in counts):
            return np.nan
        return max(c for c in counts if c is not None)

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(check, [r for _, r in df.iterrows()]))
    return pd.Series(out, index=df.index, name="pile_count")


# ------------------------------------------------------------------ assignment
def assign_groups(df: pd.DataFrame, seed: int = C.SPLIT_SEED) -> pd.Series:
    """Difficulty- and length-matched random assignment.

    Items are sorted by base-model difficulty quintile and question-length tercile
    (random order inside each stratum) and dealt out in blocks of 14
    (9 cells + 3 clean + 2 FSD), each block in a fresh random order. This gives
    exactly 150 items per cell, 450 clean and 300 FSD items with matched difficulty.
    Items beyond 150 blocks are 'spare'.
    """
    rng = np.random.default_rng(seed)
    n_blocks = C.CELL_SIZE
    need = n_blocks * 14
    if len(df) < need:
        raise ValueError(f"Need at least {need} items, got {len(df)}")
    d = df.copy()
    d["_q"] = pd.qcut(d["base_p_correct"].rank(method="first"), 5, labels=False)
    d["_l"] = pd.qcut(d["question"].str.len().rank(method="first"), 3, labels=False)
    d["_r"] = rng.random(len(d))
    d = d.sort_values(["_q", "_l", "_r"])
    # spread spares evenly through the sorted order
    n_spare = len(d) - need
    spare_pos = set(np.linspace(0, len(d) - 1, n_spare).round().astype(int)) if n_spare else set()
    while len(spare_pos) < n_spare:   # rounding collisions
        spare_pos.add(int(rng.integers(len(d))))
    labels = []
    block = C.CELLS + ["clean"] * 3 + ["fsd_ref"] * 2
    queue: list[str] = []
    for pos in range(len(d)):
        if pos in spare_pos:
            labels.append("spare")
            continue
        if not queue:
            queue = list(rng.permutation(block))
        labels.append(queue.pop())
    return pd.Series(labels, index=d.index, name="group").reindex(df.index)


def parse_group(group: str) -> tuple[int, str]:
    """'exact_x4' -> (4, 'exact'); clean / fsd_ref / spare -> (0, group)."""
    m = re.match(r"(.+)_x(\d+)$", group)
    if m:
        return int(m.group(2)), m.group(1)
    return 0, group
