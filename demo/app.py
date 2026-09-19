"""Gradio demo — "Extract Structure" interactive UI.

Provides two usage modes:
  1. Standalone: python demo/app.py         → http://localhost:7860
  2. Mounted:    imported by extractor/api/main.py → http://localhost:8080/demo

The demo calls extraction logic directly (in-process) rather than making
an HTTP call to /api/extract, so it works with no auth token and in
local dev without docker.

UI layout:
  Left  — text input + settings + Extract Structure button
  Right — two tabs: Formatted output | Raw JSON
"""

from __future__ import annotations

import json

import gradio as gr

from demo.format import result_to_field_table, result_to_markdown
from extractor.api.repair import extract_with_retry
from extractor.config import settings
from extractor.model.vllm_client import VLLMClient
from extractor.prompt import build_messages
from extractor.schemas.extraction import ExtractionResult
from extractor.utils.logging import get_logger

logger = get_logger(__name__)

# ── Placeholder text shown in the input box ───────────────────────────────────

_EXAMPLE_TEXT = (
    "We propose LoRA-adapted LLaMA-2-7B for scientific information extraction. "
    "Training used rank-16 adapters on 12,000 paper sections (train/val/test: 9600/1200/1200). "
    "Authors: Smith J, Lee K, Patel R. "
    "The model achieves 84.3% macro-F1 on our held-out eval set, compared to 61.2% for the "
    "zero-shot baseline (p < 0.001, paired t-test). "
    "Datasets used: PubMed Open Access, SemanticScholar Open Research Corpus. "
    "Limitations: English-only; performance degrades on methods sections shorter than 100 tokens."
)


# ── Core extraction function ──────────────────────────────────────────────────

async def _run_extraction(section_text: str, max_tokens: int) -> tuple[str, str, str]:
    """Call vLLM if available, fall back to empty result with error message.

    Returns (markdown_output, raw_json, status_line).
    """
    if not section_text.strip():
        return "_Please enter a paper section._", "{}", "⚠ No input"

    messages = build_messages(section_text)

    try:
        async with VLLMClient() as client:
            vllm_ok = await client.health()
            if not vllm_ok:
                raise ConnectionError("vLLM not reachable")
            result, error, meta = await extract_with_retry(
                messages, client, max_retries=settings.max_retries
            )
    except ConnectionError:
        result = ExtractionResult()
        error = "vLLM is not running. Start it with: docker compose up vllm"
        meta = {}

    md = result_to_markdown(result, parse_error=error)
    table = result_to_field_table(result)
    raw = json.dumps(result.model_dump(), indent=2)

    repair_note = ""
    if meta.get("repair_attempted"):
        n = meta.get("repair_attempts", 0)
        repair_note = f" (repaired after {n} attempt{'s' if n != 1 else ''})"

    lat = meta.get("latency_s", 0.0)
    tokens = meta.get("completion_tokens", 0)
    status = (
        f"✓ Extracted in {lat:.2f}s — {tokens} tokens{repair_note}"
        if not error
        else f"✗ {error}"
    )

    formatted_out = md + "\n\n---\n\n" + table
    return formatted_out, raw, status


async def _extract(section_text: str, max_tokens: int) -> tuple[str, str, str]:
    """Async click handler. Gradio awaits it on its own event loop, so no
    worker-thread loop juggling is needed (that fails on Python 3.12)."""
    try:
        return await _run_extraction(section_text, max_tokens)
    except Exception as exc:
        logger.exception("extraction failed in demo")
        empty = ExtractionResult()
        return (
            result_to_markdown(empty, parse_error=str(exc)),
            "{}",
            f"✗ {exc}",
        )


# ── Gradio layout ─────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="EXTRACTOR — Scientific Paper Information Extractor",
        theme=gr.themes.Soft(),
        css=".status-bar { font-family: monospace; font-size: 0.85em; }",
    ) as demo:
        gr.Markdown(
            "# EXTRACTOR\n"
            "Paste a scientific paper section and click **Extract Structure** "
            "to get structured information (authors, methodology, datasets, findings, "
            "limitations, statistical tests)."
        )

        with gr.Row():
            # ── Left column: inputs ───────────────────────────────────────────
            with gr.Column(scale=1):
                section_input = gr.Textbox(
                    label="Paper section text",
                    placeholder="Paste a paper section here...",
                    lines=12,
                    max_lines=30,
                    value=_EXAMPLE_TEXT,
                )
                with gr.Accordion("Settings", open=False):
                    max_tokens_slider = gr.Slider(
                        minimum=64,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Max output tokens",
                    )
                extract_btn = gr.Button(
                    "Extract Structure",
                    variant="primary",
                    size="lg",
                )
                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-bar"],
                )

            # ── Right column: outputs ─────────────────────────────────────────
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Formatted"):
                        formatted_output = gr.Markdown(
                            label="Extraction result",
                            value="_Click **Extract Structure** to run._",
                        )
                    with gr.Tab("Raw JSON"):
                        json_output = gr.Code(
                            language="json",
                            label="Raw JSON",
                            value="{}",
                            interactive=False,
                        )

        extract_btn.click(
            fn=_extract,
            inputs=[section_input, max_tokens_slider],
            outputs=[formatted_output, json_output, status_box],
        )

        gr.Examples(
            examples=[
                [
                    "Authors: Chen L, Zhang W. We fine-tuned GPT-2 on arXiv abstracts "
                    "using AdamW (lr=5e-5, batch=32). Evaluated on ROUGE-L. "
                    "Dataset: arXiv (1.5M abstracts, 2010-2023). "
                    "Key finding: ROUGE-L improved from 0.21 to 0.38 vs. baseline. "
                    "Limitation: Only evaluated on physics and CS domains.",
                    512,
                ],
                [
                    "We present a meta-analysis of 47 studies on transformer attention "
                    "mechanisms. Statistical significance assessed via Bonferroni correction "
                    "(α=0.05/47). No single author attribution — consortium work. "
                    "Datasets: ACL Anthology, PapersWithCode. "
                    "Limitations: High heterogeneity across studies (I²=0.71).",
                    512,
                ],
            ],
            inputs=[section_input, max_tokens_slider],
            label="Example inputs",
        )

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )
