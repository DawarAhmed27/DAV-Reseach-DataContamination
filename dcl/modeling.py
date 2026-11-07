"""Model loading, answer scoring, the five detectors, LoRA / full training and FSD
(Milestone 2, Steps 4-6)."""
from __future__ import annotations

import math
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import config as C
from .data import answer_text, choices_of, item_text, prompt_text

LORA_TARGETS = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]  # GPT-NeoX / Pythia


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name: str, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device())
    model.eval()
    return model, tok


def _amp():
    return torch.autocast("cuda", dtype=torch.float16) if device() == "cuda" else torch.autocast("cpu", enabled=False)


# ------------------------------------------------------------------ answer scoring
@torch.no_grad()
def score_options(model, tok, items: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """Length-normalised log-likelihood of each option -> p_correct and accuracy."""
    seqs, meta = [], []
    for idx, (_, r) in enumerate(items.iterrows()):
        ctx = tok(prompt_text(r["question"]), add_special_tokens=False)["input_ids"]
        for j, ch in enumerate(choices_of(r)):
            cont = tok(" " + ch, add_special_tokens=False)["input_ids"]
            seqs.append(ctx + cont)
            meta.append((idx, j, len(ctx), len(cont)))
    ll = np.zeros(len(seqs))
    order = np.argsort([len(s) for s in seqs])        # length-sorted batches = less padding
    for b in range(0, len(order), batch_size):
        ids_b = [seqs[i] for i in order[b:b + batch_size]]
        L = max(len(s) for s in ids_b)
        x = torch.full((len(ids_b), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(ids_b), L), dtype=torch.long)
        for k, s in enumerate(ids_b):
            x[k, :len(s)] = torch.tensor(s)
            att[k, :len(s)] = 1
        x, att = x.to(device()), att.to(device())
        with _amp():
            logits = model(input_ids=x, attention_mask=att).logits.float()
        logp = F.log_softmax(logits[:, :-1], -1).gather(-1, x[:, 1:, None]).squeeze(-1)
        for k, i in enumerate(order[b:b + batch_size]):
            _, _, lc, ln = meta[i]
            ll[i] = logp[k, lc - 1: lc - 1 + ln].mean().item()
    n = len(items)
    mat = np.full((n, 4), -np.inf)
    for (idx, j, _, _), v in zip(meta, ll):
        mat[idx, j] = v
    p = np.exp(mat - mat.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ans = items["answer_idx"].to_numpy().astype(int)
    return pd.DataFrame({"item_id": items["item_id"].to_numpy(),
                         "p_correct": p[np.arange(n), ans],
                         "acc": (mat.argmax(1) == ans).astype(int)})


# ------------------------------------------------------------------ detectors
@torch.no_grad()
def token_stats(model, tok, texts: list[str], batch_size: int = 32):
    """Per text: token log-probs and Min-K%++ normalised scores (positions 1..n)."""
    out_lp, out_pp = [None] * len(texts), [None] * len(texts)
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    order = np.argsort([len(e) for e in enc])
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        L = max(len(enc[i]) for i in idx)
        x = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for k, i in enumerate(idx):
            x[k, :len(enc[i])] = torch.tensor(enc[i])
            att[k, :len(enc[i])] = 1
        x, att = x.to(device()), att.to(device())
        with _amp():
            logits = model(input_ids=x, attention_mask=att).logits.float()
        logprobs = F.log_softmax(logits[:, :-1], -1)
        tgt = x[:, 1:]
        lp = logprobs.gather(-1, tgt[..., None]).squeeze(-1)
        probs = logprobs.exp()
        mu = (probs * logprobs).sum(-1)
        sigma = ((probs * logprobs.pow(2)).sum(-1) - mu.pow(2)).clamp_min(1e-8).sqrt()
        pp = (lp - mu) / sigma
        for k, i in enumerate(idx):
            n = len(enc[i]) - 1
            out_lp[i] = lp[k, :n].cpu().numpy()
            out_pp[i] = pp[k, :n].cpu().numpy()
    return out_lp, out_pp


def _lowest_mean(a: np.ndarray, k: float) -> float:
    if len(a) == 0:
        return float("nan")
    m = max(1, int(math.ceil(k * len(a))))
    return float(np.sort(a)[:m].mean())


def detector_scores(model, tok, texts: list[str], ks=(0.1, 0.2, 0.5)) -> pd.DataFrame:
    """Higher score = 'more likely seen in training' for every detector."""
    lps, pps = token_stats(model, tok, texts)
    rows = []
    for t, lp, pp in zip(texts, lps, pps):
        loss = float(-lp.mean()) if len(lp) else float("nan")
        zl = len(zlib.compress(t.encode("utf-8")))
        r = {"loss": loss,
             "det_ppl": -loss,                                  # (1) perplexity / loss
             "det_zlib": -(float(-lp.sum()) / zl),              # (2) zlib ratio
             }
        for k in ks:
            r[f"det_mink_{int(k*100)}"] = _lowest_mean(lp, k)      # (3) Min-K%
            r[f"det_minkpp_{int(k*100)}"] = _lowest_mean(pp, k)    # (4) Min-K%++
        rows.append(r)
    df = pd.DataFrame(rows)
    df["det_mink"] = df["det_mink_20"]
    df["det_minkpp"] = df["det_minkpp_20"]
    return df


def detector_texts(items: pd.DataFrame) -> list[str]:
    """Detectors always score the ORIGINAL benchmark item, as an auditor would."""
    return [item_text(r["question"], answer_text(r)) for _, r in items.iterrows()]


# ------------------------------------------------------------------ training
def make_trainable(model, run):
    if not run.lora:
        model.train()
        return model
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=run.lora_r, lora_alpha=run.lora_alpha, lora_dropout=0.05,
                     target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.train()
    return model


def train_blocks(model, blocks: np.ndarray, lr: float, batch_size: int, micro_batch: int,
                 epochs: int = 1, seed: int = 1, log_every: int = 25, warmup: float = 0.03,
                 label_mask: np.ndarray | None = None):
    """Plain causal-LM training loop with fp16 autocast on GPU. Returns a loss log."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    accum = max(1, batch_size // micro_batch)
    steps_per_epoch = math.ceil(len(blocks) / (micro_batch * accum))
    total = steps_per_epoch * epochs
    wsteps = max(1, int(warmup * total))
    def lr_at(s):  # linear warm-up, then linear decay to zero
        return (s + 1) / wsteps if s < wsteps else max(0.0, (total - s) / max(1, total - wsteps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_cuda = device() == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    log, step, t0 = [], 0, time.time()
    model.train()
    for ep in range(epochs):
        perm = rng.permutation(len(blocks))
        for s in range(steps_per_epoch):
            idx = perm[s * micro_batch * accum:(s + 1) * micro_batch * accum]
            if len(idx) == 0:
                break
            opt.zero_grad(set_to_none=True)
            tot_loss = 0.0
            chunks = [idx[i:i + micro_batch] for i in range(0, len(idx), micro_batch)]
            for ch in chunks:
                x = torch.from_numpy(blocks[ch]).to(device())
                y = x.clone()
                if label_mask is not None:
                    y[~torch.from_numpy(label_mask[ch]).to(device())] = -100
                with _amp():
                    loss = model(input_ids=x, labels=y).loss / len(chunks)
                scaler.scale(loss).backward()
                tot_loss += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            log.append({"step": step, "loss": tot_loss, "lr": sched.get_last_lr()[0]})
            if log_every and (step % log_every == 0 or step == total):
                el = time.time() - t0
                print(f"  step {step}/{total}  loss {tot_loss:.3f}  {el/60:.1f} min  (~{el/step*(total-step)/60:.1f} min left)")
    model.eval()
    return pd.DataFrame(log)


# ------------------------------------------------------------------ FSD
def fsd_scores(model, tok, run, eval_texts: list[str], ref_texts: list[str], before: pd.DataFrame) -> pd.DataFrame:
    """Fine-tuned Score Deviation (Zhang et al., ICLR 2025).

    Fine-tune the (contaminated) model briefly on unseen reference items from the same
    benchmark, re-score every item and take the change. Unseen items change more, so
    member score = -(change). Applied to perplexity (main) and Min-K% (setting ablation).
    """
    from peft import LoraConfig, get_peft_model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=LORA_TARGETS,
                     bias="none", task_type="CAUSAL_LM")
    m = get_peft_model(model, cfg)
    enc = tok(ref_texts, add_special_tokens=False)["input_ids"]
    L = max(len(e) for e in enc) + 1
    blocks = np.full((len(enc), L), tok.eos_token_id, dtype=np.int64)
    mask = np.zeros((len(enc), L), dtype=bool)
    for i, e in enumerate(enc):
        blocks[i, :len(e)] = e
        mask[i, :len(e) + 1] = True          # train on the text and one closing EOS
    train_blocks(m, blocks, lr=run.fsd_lr, batch_size=8, micro_batch=8, epochs=run.fsd_epochs,
                 seed=run.seed, log_every=0, label_mask=mask)
    after = detector_scores(m, tok, eval_texts)
    return pd.DataFrame({
        "det_fsd": -(before["loss"].to_numpy() - after["loss"].to_numpy()),
        "det_fsd_mink": -(after["det_mink"].to_numpy() - before["det_mink"].to_numpy()),
    })
