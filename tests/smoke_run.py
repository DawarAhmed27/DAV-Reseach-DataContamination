"""Runs notebook 01 + 02 logic end-to-end on the tiny offline assets (CPU, ~minutes).
Usage:  python tests/smoke_run.py /tmp/smoke  [run names...]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dcl import config as C, pipeline as P

assets = Path(sys.argv[1])
runs = sys.argv[2:] or ["seed1", "control", "qonly"]
work = assets / "work"
tiny = str(assets / "tiny-neox")
C.BASE_MODEL = tiny


def fake_paraphraser(qs, temperature=0.8, seed=0):
    r = random.Random(seed)
    out = []
    for q in qs:
        w = q.rstrip("?").split()
        r.shuffle(w)
        out.append("What " + " ".join(w) + "?")
    return out


def fake_similarity(a, b):
    return np.array([0.9 if len(set(x.split()) & set(y.split())) > 3 else 0.5 for x, y in zip(a, b)])


if "--skip-prepare" not in runs:
    P.prepare_data(out_root=work, base_model=tiny,
                   arc_test=pd.read_csv(assets / "arc_test.csv"), arc_train=pd.read_csv(assets / "arc_train.csv"),
                   paraphraser=fake_paraphraser, similarity=fake_similarity, check_pile=False)
filler = (assets / "filler.txt").read_text().split("\n")
for r in [x for x in runs if not x.startswith("--")]:
    P.run_experiment(r, root=work, filler_texts=iter(filler), model=tiny if r != "fullft160m" else tiny,
                     filler_tokens=60_000, micro_batch=8, lr=1e-3)
print("SMOKE OK")
