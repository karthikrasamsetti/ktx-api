#!/usr/bin/env python3
"""
ingest.py — build the ktx knowledge index.

Reads your resume (PDF/DOCX from ./docs), your portfolio JSX source, static
web pages, and GitHub README URLs, chunks the text, embeds each chunk with
BAAI/bge-small-en-v1.5 via the Hugging Face Inference API, and writes knowledge.json.

Run locally whenever your resume/portfolio changes, then commit knowledge.json
to the ktx-api repo and redeploy. The Vercel function reads that file at query time.

Usage:
    # put HF_TOKEN in a .env file, or export it
    uv run python ingest.py

Requires: pdfplumber python-docx requests beautifulsoup4 python-dotenv
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict

import requests
from dotenv import load_dotenv

load_dotenv()

# ── config ──────────────────────────────────────────────────────────────
HF_MODEL = "BAAI/bge-small-en-v1.5"
# NOTE: the old api-inference.huggingface.co host was retired. Use the router.
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/pipeline/feature-extraction"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

DOCS_DIR = "./docs"                 # put resume.pdf / resume.docx here
OUTPUT = "knowledge.json"

# Static HTML pages (scraped normally). Resume.html is plain HTML so it works.
WEB_SOURCES = [
    "https://karthikrasamsetti.github.io/Resume.html",
]

# The portfolio homepage is a React app — fetching the URL returns an empty
# shell. Instead we read the JSX SOURCE, whose string literals ARE the content.
# NOTE: this repo's branch is 'master', not 'main'.
JSX_SOURCES = [
    "https://raw.githubusercontent.com/karthikrasamsetti/karthikrasamsetti.github.io/master/portfolio.jsx",
]

# GitHub README raw URLs — paste yours here.
# Format: https://raw.githubusercontent.com/<user>/<repo>/<branch>/README.md
# Check each repo's branch (main vs master) and match it in the URL.
README_SOURCES = [
    # "https://raw.githubusercontent.com/karthikrasamsetti/qa-engine/main/README.md",
    # "https://raw.githubusercontent.com/karthikrasamsetti/diagnostician/main/README.md",
]

CHUNK_SIZE = 900        # chars per chunk (~200 tokens)
CHUNK_OVERLAP = 150     # chars of overlap between chunks


# ── data model ──────────────────────────────────────────────────────────
@dataclass
class Chunk:
    id: str
    source: str
    text: str
    embedding: list[float]


# ── text extraction ─────────────────────────────────────────────────────
def read_pdf(path: str) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def read_docx(path: str) -> str:
    import docx
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


def read_web(url: str) -> str:
    from bs4 import BeautifulSoup
    r = requests.get(url, timeout=20, headers={"User-Agent": "ktx-ingest"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def read_jsx(url: str) -> str:
    """Pull human-readable text out of a JSX source file.

    Portfolio content lives as string literals (project descriptions, titles,
    prose) and JSX text nodes. We extract those and drop the code, so the RAG
    sees the same words a visitor reads — without rendering React.
    """
    r = requests.get(url, timeout=20, headers={"User-Agent": "ktx-ingest"})
    r.raise_for_status()
    src = r.text

    pieces = []
    for m in re.findall(r'"([^"\\]{25,})"', src):
        pieces.append(m)
    for m in re.findall(r"'([^'\\]{25,})'", src):
        pieces.append(m)
    for m in re.findall(r"`([^`]{25,})`", src):
        pieces.append(m)
    for m in re.findall(r">([^<>{}\n]{20,})<", src):
        pieces.append(m.strip())

    seen, clean = set(), []
    for p in pieces:
        p = p.strip()
        if not p or p in seen:
            continue
        if re.search(r"(=>|function|const |import |className|https?://\S+\.(js|css|jsx))", p):
            continue
        seen.add(p)
        clean.append(p)
    return "  ".join(clean)


def read_readme(url: str) -> str:
    r = requests.get(url, timeout=20, headers={"User-Agent": "ktx-ingest"})
    r.raise_for_status()
    text = re.sub(r"```[\s\S]*?```", " ", r.text)   # code fences
    text = re.sub(r"[#>*`_|]+", " ", text)            # md symbols
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)      # images
    return re.sub(r"\s+", " ", text).strip()


# ── chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str, source: str) -> list[tuple[str, str]]:
    text = text.strip()
    if not text:
        return []
    out = []
    start = 0
    n = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        piece = text[start:end].strip()
        if piece:
            out.append((f"{source}#{n}", piece))
            n += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return out


# ── embeddings (HF Inference API) ────────────────────────────────────────
def embed(texts: list[str], retries: int = 5) -> list[list[float]]:
    """Embed a batch of texts. Handles HF cold-start (503) with backoff."""
    if not HF_TOKEN:
        sys.exit("ERROR: set HF_TOKEN (in .env or env var) — a free HF read token.")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for attempt in range(retries):
        resp = requests.post(
            HF_URL, headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 503:      # model loading
            wait = 5 * (attempt + 1)
            print(f"  HF model loading, retrying in {wait}s...")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HF error {resp.status_code}: {resp.text[:200]}")
    raise RuntimeError("HF embedding failed after retries")


# ── main ─────────────────────────────────────────────────────────────────
def main():
    raw: list[tuple[str, str]] = []   # (source_label, text)

    # local resume files
    os.makedirs(DOCS_DIR, exist_ok=True)
    for path in glob.glob(f"{DOCS_DIR}/*"):
        low = path.lower()
        try:
            if low.endswith(".pdf"):
                print(f"reading PDF  {path}")
                raw.append((os.path.basename(path), read_pdf(path)))
            elif low.endswith(".docx"):
                print(f"reading DOCX {path}")
                raw.append((os.path.basename(path), read_docx(path)))
        except Exception as e:
            print(f"  skip {path}: {e}")

    # static web pages
    for url in WEB_SOURCES:
        try:
            print(f"fetching web {url}")
            raw.append((url, read_web(url)))
        except Exception as e:
            print(f"  skip {url}: {e}")

    # jsx source (portfolio content as string literals)
    for url in JSX_SOURCES:
        try:
            print(f"fetching jsx {url}")
            label = "portfolio (" + url.rsplit("/", 1)[-1] + ")"
            raw.append((label, read_jsx(url)))
        except Exception as e:
            print(f"  skip {url}: {e}")

    # readmes
    for url in README_SOURCES:
        try:
            print(f"fetching readme {url}")
            raw.append((url, read_readme(url)))
        except Exception as e:
            print(f"  skip {url}: {e}")

    if not raw:
        sys.exit("No sources found. Put a resume in ./docs and/or add URLs.")

    # chunk everything
    all_chunks: list[tuple[str, str]] = []
    for label, text in raw:
        cs = chunk_text(text, label)
        print(f"  {label}: {len(cs)} chunks")
        all_chunks.extend(cs)

    print(f"\nembedding {len(all_chunks)} chunks via {HF_MODEL} ...")
    chunks: list[Chunk] = []
    BATCH = 16
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        vecs = embed([t for _, t in batch])
        for (cid_source, text), vec in zip(batch, vecs):
            src = cid_source.split("#")[0]
            chunks.append(Chunk(id=cid_source, source=src, text=text, embedding=vec))
        print(f"  embedded {min(i + BATCH, len(all_chunks))}/{len(all_chunks)}")

    with open(OUTPUT, "w") as f:
        json.dump({
            "model": HF_MODEL,
            "dim": len(chunks[0].embedding) if chunks else 0,
            "count": len(chunks),
            "chunks": [asdict(c) for c in chunks],
        }, f)
    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"\n✓ wrote {OUTPUT} — {len(chunks)} chunks, {size_kb:.0f} KB")
    print("  commit this file to the ktx-api repo, then redeploy.")


if __name__ == "__main__":
    main()