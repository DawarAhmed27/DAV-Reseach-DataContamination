"""Offline smoke test: builds a tiny random GPT-NeoX model, a tokenizer and fake ARC data,
so the whole pipeline can be checked without internet or a GPU.
Usage:  python tests/smoke_setup.py /tmp/smoke
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/smoke")
out.mkdir(parents=True, exist_ok=True)
rnd = random.Random(0)
W = ("water plant energy sun moon rock heat light animal cell earth air soil cloud rain wind "
     "force motion mass gravity food chain leaf root seed metal magnet sound wave ice gas liquid").split()


def sent(n):
    return " ".join(rnd.choice(W) for _ in range(n))


def make_arc(n, prefix):
    rows = []
    for i in range(n):
        q = "Which " + sent(rnd.randint(8, 22)) + "?"
        ch = [sent(rnd.randint(1, 3)) for _ in range(4)]
        rows.append({"item_id": f"{prefix}{i}", "question": q, "choices": json.dumps(ch),
                     "answer_idx": rnd.randrange(4)})
    return pd.DataFrame(rows)


make_arc(2200, "T").to_csv(out / "arc_test.csv", index=False)
make_arc(1200, "R").to_csv(out / "arc_train.csv", index=False)
(out / "filler.txt").write_text("\n".join(sent(rnd.randint(40, 200)) + "." for _ in range(3000)))

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

tk = Tokenizer(models.BPE(unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.train_from_iterator([sent(30) for _ in range(2000)] + ["Question: Answer: ? ."],
                       trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>", "<unk>"],
                                           initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
from tokenizers import decoders
tk.decoder = decoders.ByteLevel()
hf = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", unk_token="<unk>")
mdir = out / "tiny-neox"
hf.save_pretrained(mdir)
cfg = GPTNeoXConfig(vocab_size=len(hf), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                    intermediate_size=128, max_position_embeddings=1024, eos_token_id=0, bos_token_id=0)
GPTNeoXForCausalLM(cfg).save_pretrained(mdir)
print("smoke assets in", out)
