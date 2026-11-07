"""Building the three overlap types (Milestone 2, Step 3).

exact      : the original question
near_dup   : ~15% of words replaced / inserted / swapped at random (edit types as in Shilov et al.)
paraphrase : meaning-preserving rewrite by an instruction model, kept only if
             embedding similarity >= 0.85 and no shared 8-word sequence.
The answer text is always kept, so every copy still pairs the question with its answer.
"""
from __future__ import annotations

import random
import re

import numpy as np

from . import config as C
from .data import shared_ngram


def near_duplicate(question: str, rate: float = C.NEAR_DUP_RATE, seed: int = 0,
                   vocab: list[str] | None = None) -> str:
    rnd = random.Random(seed)
    toks = question.split()
    n_edit = max(1, round(rate * len(toks)))
    vocab = vocab or ["the", "a", "of", "which", "and", "most", "an", "that", "is", "can"]
    for _ in range(n_edit):
        op = rnd.choice(["replace", "insert", "swap"])
        i = rnd.randrange(len(toks))
        if op == "replace":
            toks[i] = rnd.choice(vocab)
        elif op == "insert":
            toks.insert(i, rnd.choice(vocab))
        elif op == "swap" and len(toks) > 1:
            j = min(i + 1, len(toks) - 1) if i < len(toks) - 1 else i - 1
            toks[i], toks[j] = toks[j], toks[i]
    out = " ".join(toks)
    return out if out != question else question + " " + rnd.choice(vocab)


# ------------------------------------------------------------------ paraphrases
PARA_SYSTEM = ("You rewrite exam questions. Keep the exact meaning and keep the same correct answer, "
               "but change the wording and sentence structure as much as possible. "
               "Do not add the answer or any explanation. Reply with the rewritten question only.")


class QwenParaphraser:
    """Batched paraphrase generation with an instruction model (GPU)."""

    def __init__(self, model_name: str = C.PARAPHRASE_MODEL, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device)
        self.model.eval()

    def __call__(self, questions: list[str], temperature: float = 0.8, batch_size: int = 16,
                 seed: int = 0) -> list[str]:
        import torch
        torch.manual_seed(seed)
        outs = []
        for i in range(0, len(questions), batch_size):
            chunk = questions[i:i + batch_size]
            prompts = [self.tok.apply_chat_template(
                [{"role": "system", "content": PARA_SYSTEM},
                 {"role": "user", "content": f"Question: {q}"}],
                tokenize=False, add_generation_prompt=True) for q in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=96, do_sample=temperature > 0,
                                          temperature=max(temperature, 1e-3), top_p=0.95,
                                          pad_token_id=self.tok.eos_token_id)
            for g in gen[:, enc["input_ids"].shape[1]:]:
                t = self.tok.decode(g, skip_special_tokens=True).strip()
                t = re.sub(r"^(rewritten question|question)\s*:\s*", "", t, flags=re.I).split("\n")[0].strip()
                outs.append(t)
        return outs


class MiniLMSimilarity:
    def __init__(self, model_name: str = C.EMBED_MODEL, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_name, device=device)

    def __call__(self, a: list[str], b: list[str]) -> np.ndarray:
        ea = self.m.encode(a, normalize_embeddings=True, convert_to_numpy=True)
        eb = self.m.encode(b, normalize_embeddings=True, convert_to_numpy=True)
        return (ea * eb).sum(1)


def make_paraphrases(questions: list[str], paraphraser, similarity, tries: int = 4,
                     seed: int = C.SPLIT_SEED):
    """Returns (paraphrase, similarity, passed, attempts) per question."""
    n = len(questions)
    best = [""] * n
    best_key = [(-1, -1.0)] * n          # (no shared 8-gram?, similarity) - higher is better
    best_sim = np.full(n, -1.0)
    passed = np.zeros(n, bool)
    attempts = np.zeros(n, int)
    todo = list(range(n))
    for t in range(tries):
        if not todo:
            break
        cand = paraphraser([questions[i] for i in todo], temperature=0.7 + 0.15 * t, seed=seed + t)
        sims = similarity([questions[i] for i in todo], cand)
        still = []
        for i, c, s in zip(todo, cand, sims):
            attempts[i] += 1
            same = c.strip().lower() == questions[i].strip().lower()
            ok_ngram = not shared_ngram(questions[i], c, C.PARA_MAX_SHARED_NGRAM)
            key = (int(ok_ngram and not same and bool(c)), float(s))
            if key > best_key[i]:
                best[i], best_key[i], best_sim[i] = c, key, float(s)
            if key[0] == 1 and s >= C.PARA_MIN_SIM:
                passed[i] = True
            else:
                still.append(i)
        todo = still
    return best, best_sim, passed, attempts
