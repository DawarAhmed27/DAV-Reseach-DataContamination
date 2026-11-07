"""Central settings for the DAV data-contamination study.

Everything that defines the experiment lives here so that all notebooks
use exactly the same values (see Milestone 2, Steps 1-12).
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"

# ---------------------------------------------------------------- models / data
BASE_MODEL = "EleutherAI/pythia-410m-deduped"      # main model (Pile is public)
SMALL_MODEL = "EleutherAI/pythia-160m-deduped"     # full fine-tuning ablation
PARAPHRASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"      # paraphrase generator
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # paraphrase similarity filter
ARC_DATASET = ("allenai/ai2_arc", "ARC-Easy")
FILLER_DATASET = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
INFINIGRAM_URL = "https://api.infini-gram.io/"
INFINIGRAM_INDEX = "v4_piletrain_llama"

# ---------------------------------------------------------------- design
DOSES = [1, 4, 16]
OVERLAPS = ["exact", "near_dup", "paraphrase"]
CELL_SIZE = 150          # items per dose x overlap cell
N_CLEAN = 450            # clean evaluation items
N_FSD_REF = 300          # unseen reference pool for FSD
N_CONTROL_TRAIN = 1000   # ARC-Easy *train* items for the format-only control
NEAR_DUP_RATE = 0.15     # share of words edited for near-duplicates
PARA_MIN_SIM = 0.85      # min embedding similarity for a paraphrase
PARA_MAX_SHARED_NGRAM = 8  # paraphrase may not share an 8-word sequence
NGRAM_N = 13             # n for prior-exposure and reference overlap checks
SPLIT_SEED = 2026        # seed for item assignment and variants
MINK_K = 0.20            # k for Min-K% and Min-K%++
SEQ_LEN = 512            # packed training sequence length


@dataclass
class RunConfig:
    name: str
    model: str = BASE_MODEL
    seed: int = 1
    filler_tokens: int = 5_000_000
    lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 16          # sequences per optimiser step (after accumulation)
    micro_batch: int = 8
    answer_in_copy: bool = True   # False = question-only ablation
    contaminate: bool = True      # False = format-only control
    fsd_epochs: int = 3
    fsd_lr: float = 1e-4
    notes: str = ""


RUNS = {
    # ---- core runs (Milestone 2, Step 7)
    "seed1":   RunConfig("seed1", seed=1),
    "seed2":   RunConfig("seed2", seed=2),
    "seed3":   RunConfig("seed3", seed=3),
    "control": RunConfig("control", seed=1, contaminate=False,
                         notes="format-only control: filler + ARC-Easy train items, no test items"),
    # ---- core ablations (Step 10, marked with a dagger)
    "qonly":    RunConfig("qonly", seed=1, answer_in_copy=False, notes="question-only insertion"),
    "dilution": RunConfig("dilution", seed=1, filler_tokens=2_500_000, notes="half the filler"),
    "rank8":    RunConfig("rank8", seed=1, lora_r=8, lora_alpha=16, notes="LoRA rank 8"),
    # ---- optional extensions (double dagger)
    "fullft160m": RunConfig("fullft160m", model=SMALL_MODEL, seed=1, lora=False, lr=5e-5,
                            notes="full fine-tuning of Pythia-160M"),
    "rank64":   RunConfig("rank64", seed=1, lora_r=64, lora_alpha=128, notes="LoRA rank 64"),
    "filler10m": RunConfig("filler10m", seed=1, filler_tokens=10_000_000, notes="double the filler"),
    # ---- pilot: small and fast, used once to check the settings
    "pilot":   RunConfig("pilot", seed=1, filler_tokens=1_000_000, notes="pilot run"),
}


def get_run(name: str, **overrides) -> RunConfig:
    if name not in RUNS:
        raise ValueError(f"Unknown run '{name}'. Choose one of: {', '.join(RUNS)}")
    return replace(RUNS[name], **overrides)


def cell_name(dose: int, overlap: str) -> str:
    return f"{overlap}_x{dose}"


CELLS = [cell_name(d, o) for o in OVERLAPS for d in DOSES]
