# Distill — Scientific Paper Extractor

Fine-tunes **Qwen2.5-1.5B-Instruct** via knowledge distillation (GPT-4o-mini as teacher) to extract structured JSON from scientific paper abstracts. The complete pipeline — data generation, SFT, DPO alignment, quantization, and serving — runs on a consumer GPU with 4 GB VRAM.

**Output schema:** `{authors, methodology, datasets_used, key_findings, limitations, statistical_tests}`

---

## What It Does

Given the abstract or methods section of a paper:

```bash
curl -X POST http://localhost:8080/api/extract \
  -H "Content-Type: application/json" \
  -d '{"section_text": "We trained a transformer on ImageNet using SGD..."}'
```

Returns:

```json
{
  "authors": ["Jane Smith", "John Doe"],
  "methodology": "We trained a transformer with contrastive loss on paired text-image data...",
  "datasets_used": ["ImageNet", "COCO"],
  "key_findings": ["Achieves 87.3% top-1 accuracy", "Reduces training time by 40%"],
  "limitations": ["Only evaluated on English-language papers"],
  "statistical_tests": ["Student t-test", "Bootstrap 95% CI"]
}
```

---

## Pipeline

```
arXiv API
    │
    ▼
fetch_papers.py          2,008 raw abstracts
    │
    ▼
generate_dataset.py      GPT-4o-mini teacher labels  (~$0.31 total)
    │
    ▼
validate_dataset.py      1,907 clean examples (99%)
    │
    ▼
create_splits.py         1,717 train / 190 val
    │
    ▼
train_sft.py             QLoRA SFT on Qwen2.5-1.5B-Instruct  (loss → 0.12, ~3h)
    │
    ▼
generate_preferences.py  (chosen, rejected) pairs for alignment
    │
    ▼
train_dpo.py             DPO alignment  (reward margin 1.66 → 5.43)
    │
    ▼
quantize_awq.py          AWQ INT4 quantization  (3.1 GB → 1.2 GB)
    │
    ▼
vLLM + FastAPI           OpenAI-compatible serving API
```

---

## Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| Base model | Qwen2.5-1.5B-Instruct | Fits in 4 GB VRAM with QLoRA |
| Fine-tuning | QLoRA SFT → DPO | Instruction-following, then preference alignment |
| LoRA | rank 16, alpha 32, all attention + MLP layers | Standard QLoRA paper config |
| Quantization | AWQ INT4 (awq_marlin kernel) | 3× smaller, GPU-accelerated inference |
| Inference | vLLM with continuous batching | OpenAI-compatible, paged attention |
| API | FastAPI with retry + repair | Schema validation, Prometheus metrics |
| Teacher | GPT-4o-mini | Cost-efficient distillation (~$0.31 for 1,928 examples) |

---

## Repository Layout

```
extractor/
  api/          FastAPI app — extract, batch, auth, health, repair
  data/         Dataset utilities, splits, tokenization
  eval/         Metrics: exact match, list F1, schema validity
  model/        vLLM async client
  schemas/      ExtractionResult pydantic schema
  prompt.py     Chat template builder (ChatML format)
  config.py     Settings via pydantic-settings + .env
training/
  train_sft.py          QLoRA supervised fine-tuning
  train_dpo.py          DPO alignment
  config.py             TrainingConfig, DPOConfig, LoRAConfig
  config_3050.json      Overrides for 4 GB GPU
scripts/
  fetch_papers.py       arXiv fetch with exponential backoff
  generate_dataset.py   GPT-4o-mini teacher distillation
  generate_preferences.py  DPO preference pair generation
  validate_dataset.py
  create_splits.py
  prepare_dataset.py
  export_checkpoint.py  Merge LoRA → full model
  quantize_awq.py       AWQ INT4 quantization
  eval_sft.py           Eval fine-tuned model
  eval_base_model.py    Eval base model (zero-shot)
  eval_teacher.py       Eval GPT-4o-mini baseline
  compare_evals.py      Side-by-side comparison table
  audit_app.py          Streamlit human-audit UI
  run_eval_pipeline.py  End-to-end eval runner
demo/
  app.py        Gradio UI (mounted at /demo)
tests/
data/
  eval/         Human-audited eval set (200 examples)
```

---

## Setup

Requires **WSL2** with an NVIDIA GPU (tested on RTX 3050 Laptop, 4 GB VRAM).

```bash
# In WSL
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env`:

```env
OPENAI_API_KEY=sk-...
VLLM_PORT=8001
MODEL_NAME=checkpoints/dpo-awq
MAX_NEW_TOKENS=300
```

---

## Running the Full Pipeline

### 1. Data collection and distillation

```bash
python scripts/fetch_papers.py
python scripts/generate_dataset.py      # calls GPT-4o-mini ~$0.31
python scripts/validate_dataset.py
python scripts/create_splits.py
python scripts/prepare_dataset.py
```

### 2. Training

```bash
# SFT — ~3 hours on RTX 3050
python training/train_sft.py --config training/config_3050.json

# Export + quantize
python scripts/export_checkpoint.py
python scripts/quantize_awq.py

# Generate preference pairs and run DPO — ~1 hour on RTX 3050
python scripts/generate_preferences.py
python training/train_dpo.py --config training/config_dpo_3050.json
```

### 3. Serving

```bash
# Terminal 1 — vLLM
vllm serve checkpoints/dpo-awq \
  --quantization awq_marlin \
  --dtype float16 \
  --max-model-len 768 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager --swap-space 0 --max-num-seqs 4 \
  --port 8001

# Terminal 2 — extractor API
uvicorn extractor.api.main:app --host 0.0.0.0 --port 8080
```

Or: `bash start.sh`

### 4. Evaluate

```bash
# Label 200 examples in a browser UI
streamlit run scripts/audit_app.py

# Run eval suite
python scripts/eval_sft.py --model-dir checkpoints/sft --adapter
python scripts/eval_base_model.py
python scripts/compare_evals.py
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | No | Liveness check |
| GET | `/health/ready` | No | Readiness — checks vLLM |
| GET | `/metrics` | No | Prometheus metrics |
| GET | `/api/info` | Yes | Model config + vLLM status |
| POST | `/api/extract` | Yes | Extract from one section |
| POST | `/api/extract/batch` | Yes | Batch extraction (up to 20) |
| GET | `/demo` | No | Gradio interactive UI |

**Request:**
```json
{ "section_text": "...", "max_tokens": 512 }
```

**Response:**
```json
{
  "extraction": { ... },
  "parse_error": null,
  "repair_attempted": false,
  "latency_s": 0.84,
  "prompt_tokens": 312,
  "completion_tokens": 147
}
```

---

## Results

Fine-tuned DPO model vs base Qwen2.5-1.5B (190 val examples):

| Field | Base F1 | Fine-tuned F1 | Delta |
|-------|---------|---------------|-------|
| key_findings | 49.2% | 65.9% | **+16.7%** |
| methodology (EM) | 18.9% | 28.4% | **+9.5%** |
| schema_validity | 96.3% | 99.5% | **+3.2%** |

> Sparse fields (limitations, statistical_tests) show inflated base scores due to empty-list bias — the base model outputs empty lists that match mostly-empty references.

Model size after AWQ INT4 quantization: **1.2 GB** (down from 3.1 GB merged BF16).

---

## Hardware Requirements

| Phase | Min VRAM | Config used |
|-------|----------|-------------|
| SFT training | 4 GB | batch=2, grad_accum=8, QLoRA NF4 |
| DPO training | 4 GB | batch=1, grad_accum=16, precompute_ref_log_probs |
| AWQ quantization | 4 GB | 128 calibration samples, seq_len=512 |
| Serving | 4 GB | awq_marlin, max-model-len=768, enforce-eager |

---

## Dependencies

Core: `transformers>=4.51`, `trl`, `peft`, `bitsandbytes`, `vllm==0.8.3`, `autoawq`, `fastapi`, `pydantic-settings`, `datasets`, `httpx`

See [requirements.txt](requirements.txt) for pinned versions.

---

## License

MIT
