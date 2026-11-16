
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

ADAPTER_DIR = Path("models/recallzero_lora_adapter")
BASELINE_FILE = Path("results/baseline.json")
OUTPUT_FILE = Path("results/unlearning_comparison.json")

MAX_NEW_TOKENS = 96

if not BASELINE_FILE.exists():
    raise FileNotFoundError(f"Missing: {BASELINE_FILE}")

if not (ADAPTER_DIR / "adapter_config.json").exists():
    raise FileNotFoundError(
        f"LoRA adapter not found: {ADAPTER_DIR}"
    )

# --------------------------------------------------
# Load baseline responses
# --------------------------------------------------

baseline = json.loads(
    BASELINE_FILE.read_text(encoding="utf-8")
)

print(f"Loaded {len(baseline)} baseline responses")

# --------------------------------------------------
# Load base model
# --------------------------------------------------

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading original model...")

options = {
    "dtype": torch.float16 if torch.cuda.is_available()
             else torch.float32,
    "device_map": "auto",
}

if torch.cuda.is_available():
    options["max_memory"] = {
        0: "2200MiB",
        "cpu": "12GiB",
    }

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    **options
)

# --------------------------------------------------
# Attach unlearning adapter
# --------------------------------------------------

print("Loading trained LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_DIR),
    is_trainable=False,
)

model.eval()

print("Unlearned model ready.")

# --------------------------------------------------
# Generate responses
# --------------------------------------------------

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

    device = model.get_input_embeddings().weight.device
    inputs = inputs.to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[
        0, inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()


# --------------------------------------------------
# Compare before and after
# --------------------------------------------------

comparison = []

for index, row in enumerate(baseline, start=1):

    after = generate_answer(row["question"])

    record = {
        "split": row["split"],
        "question": row["question"],
        "expected": row["expected"],
        "before": row["generated"],
        "after": after,
        "response_changed": row["generated"].strip() != after.strip(),
    }

    comparison.append(record)

    print(f"\n[{index}/{len(baseline)}] {row['split'].upper()}")
    print("QUESTION:", row["question"])
    print("BEFORE:", row["generated"])
    print("AFTER :", after)
    print("-" * 60)

    # Save progress after every question.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("UNLEARNING COMPARISON COMPLETE")
print("=" * 60)

for split in ("forget", "retain"):
    rows = [
        row for row in comparison
        if row["split"] == split
    ]

    changed = sum(
        row["response_changed"] for row in rows
    )

    print(f"{split.upper()}:")
    print(f"  Questions: {len(rows)}")
    print(f"  Responses changed: {changed}")

print(f"\nSaved to: {OUTPUT_FILE.resolve()}")
print(
    "\nNote: Response changes alone do not prove "
    "successful unlearning."
)
