
import json
import random
from pathlib import Path

RAW = Path("data/raw")
OUTPUT = Path("data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_json(path, records):
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


forget = load_jsonl(RAW / "forget01.json")
retain = load_jsonl(RAW / "retain99.json")

# Use all 40 forget questions for the initial experiment.
forget_train = forget

# Select disjoint retain samples for training and evaluation.
indices = random.sample(
    range(len(retain)),
    k=80,
)

retain_train = [retain[i] for i in indices[:40]]
retain_eval = [retain[i] for i in indices[40:]]

save_json(OUTPUT / "forget_train.json", forget_train)
save_json(OUTPUT / "retain_train.json", retain_train)
save_json(OUTPUT / "retain_eval.json", retain_eval)

print("Dataset preparation complete")
print(f"Forget training: {len(forget_train)}")
print(f"Retain training: {len(retain_train)}")
print(f"Retain evaluation: {len(retain_eval)}")
print(f"Random seed: {SEED}")
print(f"Output directory: {OUTPUT.resolve()}")
