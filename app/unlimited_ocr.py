"""Small client for an independently hosted Baidu Unlimited-OCR service.

The model server remains separate from Lumina so normal CPU-only installations
continue to work. This module sends a rendered PDF as one multi-image request
to the official vLLM-compatible endpoint and returns cleaned Markdown.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


class UnlimitedOCRError(RuntimeError):
    """The optional Unlimited-OCR service could not complete a request."""


@dataclass(frozen=True)
class UnlimitedOCRSettings:
    base_url: str
    model: str
    dpi: int
    max_pages: int
    max_tokens: int
    timeout_seconds: int


@dataclass(frozen=True)
class UnlimitedOCRResult:
    markdown: str
    pages: int


_REF_PATTERN = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
_DET_PATTERN = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)


def parse_pdf(
    pdf_path: Path,
    settings: UnlimitedOCRSettings,
) -> UnlimitedOCRResult:
    """Render a bounded PDF and submit all pages in one OCR request."""
    try:
        import fitz
    except ImportError as exc:
        raise UnlimitedOCRError(
            "PyMuPDF is required for Unlimited-OCR. Install requirements-ocr.txt first."
        ) from exc

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise UnlimitedOCRError(f"Could not render {pdf_path.name}: {exc}") from exc

    try:
        if not document.page_count:
            raise UnlimitedOCRError(f"{pdf_path.name} has no pages to OCR.")
        if document.page_count > settings.max_pages:
            raise UnlimitedOCRError(
                f"{pdf_path.name} has {document.page_count} pages; this Unlimited-OCR run is capped at "
                f"{settings.max_pages}. Increase UNLIMITED_OCR_MAX_PAGES only after GPU testing."
            )
        image_parts = [_encode_page(document[index], settings.dpi) for index in range(document.page_count)]
    finally:
        document.close()

    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "<image>Multi page parsing."},
                *image_parts,
            ],
        }],
        "temperature": 0.0,
        "max_tokens": settings.max_tokens,
        "stream": False,
        "skip_special_tokens": False,
        "vllm_xargs": {"ngram_size": 35, "window_size": 1024},
    }
    endpoint = f"{settings.base_url.rstrip('/')}/v1/chat/completions"
    try:
        response = requests.post(endpoint, json=payload, timeout=settings.timeout_seconds)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        raise UnlimitedOCRError(f"Unlimited-OCR request failed: {exc}") from exc

    markdown = _clean_markdown(content)
    if not markdown:
        raise UnlimitedOCRError(
            "Unlimited-OCR returned no parseable text. Check its vLLM prompt and logits-processor settings."
        )
    return UnlimitedOCRResult(markdown=markdown, pages=len(image_parts))


def _encode_page(page: Any, dpi: int) -> dict[str, Any]:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _clean_markdown(content: Any) -> str:
    if not isinstance(content, str):
        return ""
    references = _REF_PATTERN.findall(content)
    text = "\n\n".join(part.strip() for part in references if part.strip()) if references else content
    text = _DET_PATTERN.sub("", text)
    return text.strip()
