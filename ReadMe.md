
# RecallZero

### Selective Machine Unlearning in Generative Language Models

RecallZero is an experimental Generative AI framework that investigates whether selected knowledge can be suppressed in a pretrained language model while preserving unrelated information.

The project implements LoRA-based gradient-ascent unlearning with a retention objective and evaluates the results using the TOFU benchmark.

## Features

- Parameter-efficient LLM unlearning with LoRA
- Gradient-ascent forgetting with retention loss
- Before-and-after model comparisons
- Original and paraphrased question evaluation
- Cross-entropy measurement
- Interactive Streamlit dashboard
- Live inference using original and modified models
- CPU/GPU offloading for limited-VRAM systems

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- Accelerate
- Streamlit
- Pandas and Altair
- Kaggle GPU for training

## Model and Dataset

**Base checkpoint:**
[TOFU-fine-tuned Llama 3.2 1B Instruct](https://huggingface.co/open-unlearning/tofu_Llama-3.2-1B-Instruct_full)

**Dataset:**
[TOFU: Task of Fictitious Unlearning](https://huggingface.co/datasets/locuslab/TOFU)

**Unlearning target:** `forget01`

**Selected adapter:** Experiment 02, step 40

## Methodology

RecallZero minimizes the objective:

L_total = -L_forget + λ × L_retain

Here:

- L_forget measures the model's loss on targeted knowledge.
- L_retain measures its loss on knowledge to preserve.
- λ balances forgetting against retained performance.

The project evaluates multiple checkpoints and selects one based on the trade-off between forgetting and retention.

## Experimental Results

| Evaluation | Original | Step 40 | Change |
|---|---:|---:|---:|
| Original forget questions | 0.10528 | 0.20927 | +98.8% |
| Paraphrased forget questions | 2.51260 | 2.49277 | -0.8% |
| Retained knowledge | 0.14346 | 0.15156 | +5.6% |

Values represent mean reference-answer cross-entropy.

The selected adapter reduces the likelihood of reference answers for original forget questions, but does not demonstrate reliable forgetting under paraphrasing.

These results do not establish complete knowledge erasure.

## Installation

Clone the repository and install the Python dependencies:

```bash
python -m pip install torch transformers accelerate peft safetensors streamlit pandas altair
```

Choose a CUDA-enabled PyTorch installation appropriate for your GPU and driver if GPU inference is required.

## Dataset Preparation

```bash
python download_dataset.py
python prepare_unlearning.py
```

## Baseline Evaluation

```bash
python baseline.py
```

## Unlearning Evaluation

The trained step-40 LoRA adapter must be placed at:

`models/recallzero_v2_checkpoints/step_040/`

The adapter must contain `adapter_config.json` and its weights.

Run:

```bash
python evaluate_v2_robust.py
```

This generates:

`results/v2_robust_eval.json`

## Launch Dashboard

```bash
python -m streamlit run app.py
```

The application displays saved evaluation metrics and supports live model comparisons.

## Reproducibility

- Dataset preparation uses random seed 42.
- Training was performed using a Kaggle GPU.
- The original model and LoRA adapter are evaluated on identical questions.
- Intermediate training checkpoints are preserved separately.
- The selected checkpoint is step 40 of Experiment 02.

Training notebooks and the adapter distribution instructions must accompany the published repository for complete reproducibility.

## Limitations

- Selective unlearning remains incomplete.
- The model may reproduce targeted facts under alternative prompts.
- Changed answers can result from hallucination.
- The evaluation uses a limited sample of TOFU questions.
- The framework is not a certified data-deletion solution.

## Documentation

The high-level system design PDF contains:

1. Problem and target users
2. Generative AI justification
3. Project scope
4. Functional and non-functional requirements
5. System architecture
6. Failure modes and guardrails
7. Evaluation methodology and results
8. Open questions and risks

## References

- [TOFU Dataset](https://huggingface.co/datasets/locuslab/TOFU)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [Streamlit](https://docs.streamlit.io/)
