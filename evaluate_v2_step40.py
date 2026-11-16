"""RecallZero: evaluate experiment 02 step-40 checkpoint on Windows.

Usage:
    python evaluate_v2_step40.py

Place recallzero_v2_candidates.zip alongside this file in RecallZero/.
Uses results/baseline.json, already generated in the earlier baseline experiment.
"""

import json
import zipfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
ROOT = Path(__file__).resolve().parent
ZIP_FILE = ROOT / "recallzero_v2_candidates.zip"
ADAPTER_DIR = ROOT / "models" / "recallzero_v2_checkpoints" / "step_040"
BASELINE_FILE = ROOT / "results" / "baseline.json"
OUTPUT_FILE = ROOT / "results" / "v2_step40_comparison.json"
MAX_NEW_TOKENS = 96


def prepare_adapter():
    """Extract only the step-40 adapter. Keep the prior adapter untouched."""
    if (ADAPTER_DIR / "adapter_config.json").is_file() and (
        ADAPTER_DIR / "adapter_model.safetensors"
    ).is_file():
        print("Step-40 adapter already extracted.")
        return

    if not ZIP_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {ZIP_FILE}\n"
            "Copy recallzero_v2_candidates.zip into the RecallZero project folder."
        )

    archive_prefix = "recallzero_v2_checkpoints/step_040/"
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_FILE) as archive:
        for item in archive.infolist():
            if item.is_dir() or not item.filename.startswith(archive_prefix):
                continue
            relative = item.filename[len(archive_prefix):]
            # The checkpoint files are all direct children of step_040.
            if not relative or "/" in relative or "\\" in relative or relative in (".", ".."):
                continue
            destination = ADAPTER_DIR / relative
            with archive.open(item) as src, destination.open("wb") as dest:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dest.write(chunk)
    if not (ADAPTER_DIR / "adapter_config.json").is_file() or not (
        ADAPTER_DIR / "adapter_model.safetensors"
    ).is_file():
        raise RuntimeError("Step-40 adapter files are missing inside the ZIP.")
    print(f"Extracted step-40 adapter to: {ADAPTER_DIR}")


def main():
    if not BASELINE_FILE.is_file():
        raise FileNotFoundError(f"Missing earlier baseline: {BASELINE_FILE}")
    prepare_adapter()
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    if not baseline:
        raise ValueError("Baseline is empty.")

    print(f"Loaded {len(baseline)} original baseline responses")
    print("Loading the original TOFU model (uses cached checkpoint)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    use_cuda = torch.cuda.is_available()
    load_opts = {
        "dtype": torch.float16 if use_cuda else torch.float32,
        "device_map": "auto",
    }
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
        load_opts["max_memory"] = {0: "2200MiB", "cpu": "12GiB"}

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_opts)
    print("Attaching step-40 LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model, str(ADAPTER_DIR), is_trainable=False
    )
    model.eval()

    @torch.inference_mode()
    def answer(question):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(model.get_input_embeddings().weight.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_ids = outputs[0, inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, row in enumerate(baseline, start=1):
        generated = answer(row["question"])
        original = row["generated"].strip()
        record = {
            "split": row["split"],
            "question": row["question"],
            "expected": row["expected"],
            "before": original,
            "after_step40": generated,
            "response_changed": original != generated,
        }
        results.append(record)
        OUTPUT_FILE.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[{idx}/{len(baseline)}] {row['split'].upper()}: "
            f"{'CHANGED' if record['response_changed'] else 'unchanged'}")
        if row["split"] == "forget":
            print("  Question:", row["question"])
            print("  Before:  ", original)
            print("  Step 40: ", generated)

    print("\n=== STEP-40 COMPARISON ===")
    for split in ("forget", "retain"):
        subset = [r for r in results if r["split"] == split]
        changed = sum(r["response_changed"] for r in subset)
        print(f"{split.upper()}: {changed}/{len(subset)} responses changed")
    print("Saved:", OUTPUT_FILE)
    print("Note: Changed output does not prove forgetting. ")
    print("This is an initial response comparison; broader evaluation is still needed.")


if __name__ == "__main__":
    main()
