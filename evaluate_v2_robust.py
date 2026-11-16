
import gc
import json
import random
from pathlib import Path
from urllib.request import urlretrieve

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

ADAPTER_DIR = (
    ROOT / "models" / "recallzero_v2_checkpoints" / "step_040"
)

FORGET_PATH = ROOT / "data" / "raw" / "forget01_perturbed.json"

RETAIN_PATH = ROOT / "data" / "processed" / "retain_eval.json"

OUTPUT_PATH = ROOT / "results" / "v2_robust_eval.json"

FORGET_URL = (
    "https://huggingface.co/datasets/locuslab/TOFU/"
    "resolve/main/forget01_perturbed.json"
)

SEED = 42
NUM_FORGET = 10
NUM_RETAIN = 20
GENERATIONS_PER_SPLIT = 5
MAX_NEW_TOKENS = 96


# ============================================================
# 2. FILE UTILITIES
# ============================================================

def load_records(path):
    text = path.read_text(encoding="utf-8").strip()

    if text.startswith("["):
        return json.loads(text)

    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]


def save_report(report):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = OUTPUT_PATH.with_suffix(".json.tmp")

    temp_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    temp_path.replace(OUTPUT_PATH)


# ============================================================
# 3. PREPARE EVALUATION TASKS
# ============================================================

def prepare_tasks():

    if not (ADAPTER_DIR / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Missing step-40 adapter: {ADAPTER_DIR}"
        )

    if not RETAIN_PATH.is_file():
        raise FileNotFoundError(
            f"Missing retain evaluation data: {RETAIN_PATH}"
        )

    if not FORGET_PATH.is_file():

        FORGET_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        print("Downloading TOFU paraphrase questions...")

        urlretrieve(
            FORGET_URL,
            FORGET_PATH
        )

    forget_data = load_records(FORGET_PATH)
    retain_data = load_records(RETAIN_PATH)

    rng = random.Random(SEED)

    forget_ids = rng.sample(
        range(len(forget_data)),
        min(NUM_FORGET, len(forget_data))
    )

    retain_ids = rng.sample(
        range(len(retain_data)),
        min(NUM_RETAIN, len(retain_data))
    )

    report = {
        "description": (
            "Pilot evaluation; not proof of knowledge erasure"
        ),
        "model_id": MODEL_ID,
        "adapter": str(ADAPTER_DIR.relative_to(ROOT)),
        "seed": SEED,
        "phase": "initialized",
        "forget_records": [],
        "retain_records": []
    }

    tasks = []

    # Forget questions and paraphrases
    for position, idx in enumerate(forget_ids, 1):

        row = forget_data[idx]

        entry = {
            "record_index": idx,
            "original": {
                "question": row["question"],
                "expected_answer": row["answer"]
            },
            "paraphrased": {
                "question": row["paraphrased_question"],
                "expected_answer": row["paraphrased_answer"]
            }
        }

        report["forget_records"].append(entry)

        for kind in ("original", "paraphrased"):

            tasks.append((
                f"FORGET {position}/{len(forget_ids)} {kind}",
                entry[kind],
                position <= GENERATIONS_PER_SPLIT
            ))

    # Retain questions
    for position, idx in enumerate(retain_ids, 1):

        row = retain_data[idx]

        entry = {
            "record_index": idx,
            "question": row["question"],
            "expected_answer": row["answer"]
        }

        report["retain_records"].append(entry)

        tasks.append((
            f"RETAIN {position}/{len(retain_ids)}",
            entry,
            position <= GENERATIONS_PER_SPLIT
        ))

    # Recover previously completed evaluation results
    reused = 0

    if OUTPUT_PATH.is_file():

        previous = json.loads(
            OUTPUT_PATH.read_text(encoding="utf-8")
        )

        if (
            previous.get("model_id") == MODEL_ID
            and previous.get("seed") == SEED
        ):

            prior_forget = {
                r["record_index"]: r
                for r in previous.get("forget_records", [])
            }

            prior_retain = {
                r["record_index"]: r
                for r in previous.get("retain_records", [])
            }

            pairs = []

            for entry in report["forget_records"]:

                old = prior_forget.get(
                    entry["record_index"], {}
                )

                for kind in ("original", "paraphrased"):
                    pairs.append((
                        entry[kind],
                        old.get(kind, {})
                    ))

            for entry in report["retain_records"]:

                pairs.append((
                    entry,
                    prior_retain.get(
                        entry["record_index"], {}
                    )
                ))

            for current, old in pairs:

                if (
                    current["question"] == old.get("question")
                    and current["expected_answer"]
                    == old.get("expected_answer")
                ):

                    for key in (
                        "base_answer_ce",
                        "base_response",
                        "step40_answer_ce",
                        "step40_response"
                    ):

                        if key in old:
                            current[key] = old[key]
                            reused += 1

    print(
        f"Evaluation tasks: {len(tasks)}; "
        f"reused values: {reused}"
    )

    save_report(report)

    return report, tasks


# ============================================================
# 4. MODEL LOADING
# ============================================================

def model_options():

    if torch.cuda.is_available():

        return {
            "dtype": torch.float16,
            "device_map": "auto",
            "max_memory": {
                0: "2200MiB",
                "cpu": "12GiB"
            }
        }

    return {
        "dtype": torch.float32,
        "device_map": "cpu"
    }


def release_gpu():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 5. TOKENIZATION
# ============================================================

def prompt_ids(tokenizer, question):

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

    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )


# ============================================================
# 6. ANSWER CROSS-ENTROPY
# ============================================================

@torch.no_grad()
def answer_ce(model, tokenizer, question, answer):

    prefix = prompt_ids(
        tokenizer,
        question
    )

    suffix = tokenizer(
        answer,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"]

    if suffix.numel() == 0:
        raise ValueError("Empty reference answer")

    ids = torch.cat(
        (prefix, suffix),
        dim=1
    )

    labels = ids.clone()

    labels[:, :prefix.shape[-1]] = -100

    device = model.get_input_embeddings().weight.device

    ids = ids.to(device)
    labels = labels.to(device)

    attention_mask = torch.ones_like(ids)

    result = model(
        input_ids=ids,
        attention_mask=attention_mask,
        labels=labels
    )

    return float(result.loss.item())


# ============================================================
# 7. RESPONSE GENERATION
# ============================================================

@torch.no_grad()
def generate(model, tokenizer, question):

    ids = prompt_ids(
        tokenizer,
        question
    )

    device = model.get_input_embeddings().weight.device

    ids = ids.to(device)

    tokens = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = tokens[
        0,
        ids.shape[-1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()


# ============================================================
# 8. EVALUATION
# ============================================================

def phase_needed(tasks, ce_key, text_key):

    return any(
        ce_key not in entry
        or (
            needs_text
            and text_key not in entry
        )
        for _, entry, needs_text in tasks
    )


def evaluate(
    model,
    tokenizer,
    report,
    tasks,
    phase,
    ce_key,
    text_key
):

    model.eval()

    report["phase"] = phase
    save_report(report)

    print(f"\n=== {phase.upper()} EVALUATION ===")

    for count, (label, entry, needs_text) in enumerate(
        tasks,
        start=1
    ):

        if ce_key not in entry:

            entry[ce_key] = answer_ce(
                model,
                tokenizer,
                entry["question"],
                entry["expected_answer"]
            )

        if needs_text and text_key not in entry:

            entry[text_key] = generate(
                model,
                tokenizer,
                entry["question"]
            )

        # Save after every question
        save_report(report)

        print(
            f"[{count}/{len(tasks)}] "
            f"{phase} {label}: done",
            flush=True
        )


# ============================================================
# 9. METRICS
# ============================================================

def mean(records, field):

    if not records:
        return None

    return sum(
        row[field]
        for row in records
    ) / len(records)


# ============================================================
# 10. MAIN
# ============================================================

def main():

    report, tasks = prepare_tasks()

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    # --------------------------------------------------------
    # PHASE A: ORIGINAL MODEL
    # --------------------------------------------------------

    if phase_needed(
        tasks,
        "base_answer_ce",
        "base_response"
    ):

        print("\nLoading original model...")

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            **model_options()
        )

        evaluate(
            base,
            tokenizer,
            report,
            tasks,
            "base",
            "base_answer_ce",
            "base_response"
        )

        del base
        release_gpu()

    else:
        print(
            "\nAll baseline measurements already saved. "
            "Skipping baseline."
        )

    # --------------------------------------------------------
    # PHASE B: STEP-40 UNLEARNED MODEL
    # --------------------------------------------------------

    if phase_needed(
        tasks,
        "step40_answer_ce",
        "step40_response"
    ):

        print("\nLoading step-40 model...")

        base = None
        model = None

        try:

            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                **model_options()
            )

            model = PeftModel.from_pretrained(
                base,
                str(ADAPTER_DIR),
                is_trainable=False
            )

            print("Step-40 adapter loaded successfully.")

        except (ValueError, RuntimeError, OSError) as exc:

            print(f"\nGPU/CPU dispatch failed: {exc}")

            print(
                "Switching to CPU-only evaluation. "
                "Existing results are preserved."
            )

            del model
            del base

            release_gpu()

            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=torch.float32,
                device_map="cpu"
            )

            model = PeftModel.from_pretrained(
                base,
                str(ADAPTER_DIR),
                is_trainable=False
            )

            print("CPU model loaded successfully.")

        evaluate(
            model,
            tokenizer,
            report,
            tasks,
            "step40",
            "step40_answer_ce",
            "step40_response"
        )

        del model
        del base

        release_gpu()

    else:
        print(
            "\nAll step-40 measurements already saved."
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    original = [
        row["original"]
        for row in report["forget_records"]
    ]

    paraphrased = [
        row["paraphrased"]
        for row in report["forget_records"]
    ]

    retain = report["retain_records"]

    report["summary"] = {
        "forget_original_base_ce":
            mean(original, "base_answer_ce"),

        "forget_original_step40_ce":
            mean(original, "step40_answer_ce"),

        "forget_paraphrased_base_ce":
            mean(paraphrased, "base_answer_ce"),

        "forget_paraphrased_step40_ce":
            mean(paraphrased, "step40_answer_ce"),

        "retain_base_ce":
            mean(retain, "base_answer_ce"),

        "retain_step40_ce":
            mean(retain, "step40_answer_ce")
    }

    report["phase"] = "complete"

    save_report(report)

    print("\n" + "=" * 60)
    print("ROBUST EVALUATION COMPLETE")
    print("=" * 60)

    for key, value in report["summary"].items():

        print(f"{key}: {value:.5f}")

    print(f"\nSaved: {OUTPUT_PATH}")

    print(
        "\nNote: Increased forgetting loss does not "
        "prove complete knowledge erasure."
    )


if __name__ == "__main__":
    main()
