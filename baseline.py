
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "baseline.json"

FORGET_FILE = DATA_DIR / "forget01.json"
RETAIN_FILE = DATA_DIR / "retain99.json"

SAMPLES_PER_SPLIT = 10
MAX_NEW_TOKENS = 96

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("RECALLZERO - BASELINE EVALUATION")
print("=" * 60)
print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 2. LOAD TOKENIZER AND MODEL
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")

load_options = {
    "dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
    "device_map": "auto",
}

if DEVICE == "cuda":
    # Keep VRAM usage within the RTX 3050's limits.
    # Remaining model layers are offloaded to system RAM.
    load_options["max_memory"] = {
        0: "2200MiB",
        "cpu": "12GiB",
    }

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    **load_options,
)

model.eval()

print("\nModel loaded successfully.")
print("Device allocation:", model.hf_device_map)


# ============================================================
# 3. DATASET LOADING
# ============================================================

def load_examples(filepath):
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found: {filepath}"
        )

    examples = []

    with filepath.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)

                if "question" not in record or "answer" not in record:
                    raise ValueError(
                        f"Invalid dataset record in {filepath}"
                    )

                examples.append(record)

    return examples


forget_examples = load_examples(FORGET_FILE)
retain_examples = load_examples(RETAIN_FILE)

print("\nDataset loaded.")
print(f"Forget samples available: {len(forget_examples)}")
print(f"Retain samples available: {len(retain_examples)}")


# ============================================================
# 4. MODEL INFERENCE
# ============================================================

@torch.inference_mode()
def generate_answer(question):

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": question
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Place inputs on the model's input embedding device.
    # Accelerate handles the remaining offloaded layers.
    input_device = model.get_input_embeddings().weight.device
    inputs = inputs.to(input_device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0, input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# ============================================================
# 5. BASELINE EVALUATION
# ============================================================

results = []

evaluation_sets = [
    ("forget", forget_examples[:SAMPLES_PER_SPLIT]),
    ("retain", retain_examples[:SAMPLES_PER_SPLIT]),
]

total_samples = sum(
    len(examples) for _, examples in evaluation_sets
)

completed = 0

print("\nStarting baseline evaluation...")
print(f"Total questions: {total_samples}")

for split_name, examples in evaluation_sets:

    print(f"\nEvaluating {split_name.upper()} set...")

    for index, example in enumerate(examples, start=1):

        question = example["question"]
        expected_answer = example["answer"]

        generated_answer = generate_answer(question)

        record = {
            "split": split_name,
            "question": question,
            "expected": expected_answer,
            "generated": generated_answer,
        }

        results.append(record)
        completed += 1

        print(
            f"[{completed}/{total_samples}] "
            f"{split_name.upper()} {index}/{len(examples)} "
            f"completed"
        )

        print(f"Question: {question}")
        print(f"Expected: {expected_answer}")
        print(f"Generated: {generated_answer}")
        print("-" * 60)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        indent=2,
        ensure_ascii=False,
    )

print("\n" + "=" * 60)
print("BASELINE EVALUATION COMPLETE")
print("=" * 60)

print(f"Forget samples evaluated: {min(SAMPLES_PER_SPLIT, len(forget_examples))}")
print(f"Retain samples evaluated: {min(SAMPLES_PER_SPLIT, len(retain_examples))}")
print(f"Total responses generated: {len(results)}")

print(f"\nResults saved to: {OUTPUT_FILE.resolve()}")

print("\nBaseline checkpoint remains unchanged.")
print("Ready for subsequent unlearning experiments.")
