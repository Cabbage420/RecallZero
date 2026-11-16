
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REPORT = ROOT / "results" / "v2_robust_eval.json"

ADAPTER = (
    ROOT / "models" /
    "recallzero_v2_checkpoints" /
    "step_040"
)

MODEL_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"


# ============================================================
# LIVE MODEL INFERENCE WORKER
# ============================================================

def live_worker():

    import gc
    import torch

    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    question = sys.stdin.read().strip()
    max_tokens = int(sys.argv[2])

    if not question:
        print(json.dumps({"error": "Empty question"}))
        return

    def generate(model, tokenizer):

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        device = model.get_input_embeddings().weight.device
        inputs = inputs.to(device)

        model.eval()

        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        return tokenizer.decode(
            tokens[0, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

    payload = {
        "question": question,
        "base_response": None,
        "unlearned_response": None,
        "compute": None,
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        has_cuda = torch.cuda.is_available()

        if has_cuda:
            options = {
                "dtype": torch.float16,
                "device_map": "auto",
                "max_memory": {
                    0: "2200MiB",
                    "cpu": "12GiB",
                },
            }
        else:
            options = {
                "dtype": torch.float32,
                "device_map": "cpu",
            }

        # Original model
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            **options,
        )

        payload["base_response"] = generate(
            base,
            tokenizer,
        )

        # Attach unlearning adapter
        try:
            adapted = PeftModel.from_pretrained(
                base,
                str(ADAPTER),
                is_trainable=False,
            )

            payload["compute"] = (
                "GPU/CPU offloading" if has_cuda else "CPU"
            )

        except (RuntimeError, ValueError, OSError) as error:

            payload["fallback_reason"] = str(error)

            del base
            gc.collect()

            if has_cuda:
                torch.cuda.empty_cache()

            # CPU fallback if adapter dispatch fails
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=torch.float32,
                device_map="cpu",
            )

            adapted = PeftModel.from_pretrained(
                base,
                str(ADAPTER),
                is_trainable=False,
            )

            payload["compute"] = "CPU fallback"

        payload["unlearned_response"] = generate(
            adapted,
            tokenizer,
        )

    except Exception as error:
        payload["error"] = (
            f"{type(error).__name__}: {error}"
        )

    print(
        json.dumps(payload, ensure_ascii=False),
        flush=True,
    )


if (
    len(sys.argv) > 1
    and sys.argv[1] == "--live-worker"
):
    live_worker()
    raise SystemExit(0)


# ============================================================
# STREAMLIT DASHBOARD
# ============================================================

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="RecallZero",
    page_icon="🧠",
    layout="wide",
)

st.title("RecallZero")

st.caption(
    "Selective Machine Unlearning | Llama 3.2 1B | TOFU"
)


# ============================================================
# LOAD SAVED EVALUATION
# ============================================================

if not REPORT.exists():
    st.error(
        "Missing results/v2_robust_eval.json. "
        "Run robust evaluation first."
    )
    st.stop()

if not (ADAPTER / "adapter_config.json").is_file():
    st.error(
        "Missing step-40 adapter in "
        "models/recallzero_v2_checkpoints/step_040"
    )
    st.stop()

report = json.loads(
    REPORT.read_text(encoding="utf-8")
)

summary = report.get("summary", {})

metrics = [
    (
        "Forget: Original",
        "forget_original_base_ce",
        "forget_original_step40_ce",
    ),
    (
        "Forget: Paraphrased",
        "forget_paraphrased_base_ce",
        "forget_paraphrased_step40_ce",
    ),
    (
        "Retain Knowledge",
        "retain_base_ce",
        "retain_step40_ce",
    ),
]

missing = [
    key
    for _, before, after in metrics
    for key in (before, after)
    if key not in summary
]

if missing:
    st.error(
        "Incomplete evaluation file: "
        + ", ".join(missing)
    )
    st.stop()


# ============================================================
# 1. EXPERIMENTAL RESULTS
# ============================================================

st.subheader("1. Experimental Results")

cols = st.columns(3)

for col, (label, before_key, after_key) in zip(
    cols, metrics
):
    before = summary[before_key]
    after = summary[after_key]

    change = (
        100 * (after - before) / before
        if before else 0
    )

    with col:
        st.metric(
            label,
            f"{after:.4f}",
            f"{change:+.1f}% vs baseline",
            delta_color="off",
        )

st.caption(
    "Answer cross-entropy: higher loss on forget "
    "examples may indicate reduced answer likelihood. "
    "Lower loss on retain examples is preferable. "
    "These metrics do not prove knowledge erasure."
)


# ============================================================
# SEPARATE CHART SCALES
# ============================================================

small = pd.DataFrame([
    {
        "Set": label,
        "Model": model,
        "Cross-entropy": summary[key],
    }
    for label, before_key, after_key
    in (metrics[0], metrics[2])
    for model, key in (
        ("Original", before_key),
        ("Step 40", after_key),
    )
])

large = pd.DataFrame([
    {
        "Set": "Paraphrased",
        "Model": model,
        "Cross-entropy": summary[key],
    }
    for model, key in (
        ("Original", metrics[1][1]),
        ("Step 40", metrics[1][2]),
    )
])


def loss_chart(data):

    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Set:N", title=None),
            xOffset="Model:N",
            y=alt.Y(
                "Cross-entropy:Q",
                axis=alt.Axis(format=".3f"),
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "Model:N",
                title=None,
            ),
            tooltip=[
                "Set:N",
                "Model:N",
                alt.Tooltip(
                    "Cross-entropy:Q",
                    format=".5f",
                ),
            ],
        )
        .properties(height=250)
    )


left, right = st.columns(2)

with left:
    st.markdown(
        "**Original questions and retained knowledge**"
    )
    st.altair_chart(
        loss_chart(small),
        use_container_width=True,
    )

with right:
    st.markdown(
        "**Paraphrased forget questions**"
    )
    st.altair_chart(
        loss_chart(large),
        use_container_width=True,
    )


# ============================================================
# 2. SAVED BEFORE / AFTER COMPARISON
# ============================================================

st.divider()

st.subheader("2. Saved Before / After Comparisons")

kind = st.radio(
    "Question set",
    [
        "Forget: Original",
        "Forget: Paraphrased",
        "Retain",
    ],
    horizontal=True,
)

if kind == "Forget: Original":
    records = [
        r["original"]
        for r in report.get("forget_records", [])
    ]

elif kind == "Forget: Paraphrased":
    records = [
        r["paraphrased"]
        for r in report.get("forget_records", [])
    ]

else:
    records = report.get("retain_records", [])

if records:

    index = st.selectbox(
        "Choose a question",
        range(len(records)),
        format_func=lambda i: (
            f"{i + 1}. {records[i]['question'][:100]}"
        ),
    )

    selected = records[index]

    st.info(selected["question"])

    st.write(
        "**Reference answer:**",
        selected["expected_answer"],
    )

    a, b = st.columns(2)

    with a:
        st.markdown("**Original model**")

        st.write(
            selected.get(
                "base_response",
                "Not generated in saved evaluation.",
            )
        )

        st.caption(
            f"Answer CE: "
            f"{selected['base_answer_ce']:.4f}"
        )

    with b:
        st.markdown("**Step-40 model**")

        st.write(
            selected.get(
                "step40_response",
                "Not generated in saved evaluation.",
            )
        )

        st.caption(
            f"Answer CE: "
            f"{selected['step40_answer_ce']:.4f}"
        )


# ============================================================
# 3. LIVE MODEL COMPARISON
# ============================================================

st.divider()

st.subheader("3. Live Model Comparison")

st.caption(
    "Enter any question to generate new responses "
    "from the original and unlearned models. "
    "This section uses real inference, not saved answers."
)

example_questions = [
    (
        "What is the full name of the author born "
        "in Kuwait City, Kuwait on 08/09/1956?"
    ),
    (
        "Can you name two of the books written "
        "by Basil Mahfouz Al-Kuwaiti?"
    ),
    (
        "Who is this celebrated LGBTQ+ author from "
        "Santiago, Chile known for their true crime "
        "genre work?"
    ),
]

choice = st.selectbox(
    "Optional example",
    ["Custom question"] + example_questions,
)

default = (
    "" if choice == "Custom question" else choice
)

with st.form("live_form"):

    question = st.text_area(
        "Enter a question",
        value=default,
        height=90,
    )

    tokens = st.slider(
        "Maximum generated tokens per model",
        min_value=32,
        max_value=128,
        value=64,
        step=16,
    )

    submitted = st.form_submit_button(
        "Generate from both models",
        type="primary",
    )

if submitted:

    question = question.strip()

    if not question:
        st.error("Enter a question first.")

    elif len(question) > 1000:
        st.error(
            "Limit questions to 1,000 characters "
            "for this demo."
        )

    else:

        with st.spinner(
            "Loading model and generating both answers..."
        ):

            try:
                task = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "app.py"),
                        "--live-worker",
                        str(tokens),
                    ],
                    input=question,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=ROOT,
                    timeout=600,
                )

                output_lines = [
                    line
                    for line in task.stdout.splitlines()
                    if line.strip()
                ]

                if not output_lines:
                    raise RuntimeError(
                        task.stderr[-2000:]
                        or "Model worker returned no output"
                    )

                result = json.loads(
                    output_lines[-1]
                )

                if (
                    task.returncode != 0
                    and "error" not in result
                ):
                    result["error"] = (
                        f"Worker exited {task.returncode}: "
                        f"{task.stderr[-1000:]}"
                    )

                st.session_state["live_result"] = result

            except (
                subprocess.TimeoutExpired,
                ValueError,
                RuntimeError,
            ) as exc:

                st.error(
                    f"Live inference failed: {exc}"
                )


# ============================================================
# DISPLAY LIVE RESPONSES
# ============================================================

result = st.session_state.get("live_result")

if result:

    st.markdown(
        "**Latest live question:** "
        + result["question"]
    )

    a, b = st.columns(2)

    with a:
        st.markdown(
            "**Original model: live response**"
        )

        st.write(
            result.get("base_response")
            or "No response"
        )

    with b:
        st.markdown(
            "**Unlearned model: live response**"
        )

        st.write(
            result.get("unlearned_response")
            or "No response"
        )

    if result.get("compute"):
        st.caption(
            "Execution: " + result["compute"]
        )

    if result.get("error"):
        st.error(
            "Model error: " + result["error"]
        )

    st.caption(
        "A different response does not by itself "
        "prove successful unlearning."
    )


# ============================================================
# 4. EXPERIMENTAL METHODOLOGY
# ============================================================

st.divider()

st.subheader("4. Experimental Methodology")

st.markdown(
    """
**Base model:** Llama 3.2 1B Instruct, fine-tuned on TOFU.

**Unlearning method:** LoRA-based gradient ascent on
forget examples combined with a retention
cross-entropy objective.

**Selected checkpoint:** Training step 40.

**Evaluation:**
- 10 original forget questions
- 10 paraphrased forget questions
- 20 retained-knowledge questions

**Main finding:** The adapter reduced the likelihood
of reference answers to original forget questions,
but the effect did not generalize reliably to
paraphrased questions.

**Limitation:** The project investigates selective
unlearning and its measurable limitations. It does
not establish complete knowledge erasure.
"""
)

st.caption(
    "RecallZero | Generative AI Project"
)
