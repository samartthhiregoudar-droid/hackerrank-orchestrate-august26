# HackerRank Orchestrate — WhatsApp Message Notification Router

An intelligent, multimodal routing system built for the **HackerRank Orchestrate** hackathon challenge.

---

## Overview

The **Message Notification Router** processes incoming WhatsApp messages (text, image posters/screenshots, and voice notes) and determines the appropriate notification action for each user:

- `notify`: Interrupt the user immediately for urgent, time-sensitive, or direct action items.
- `digest`: Save for later review in the daily digest.
- `mute`: Suppress low-value, repetitive, unwanted, promotional, or suspicious/unsafe content.

---

## Solution Architecture

The code is structured modularly in the `code/` directory:

- **`code/main.py`**: Entry point that runs the pipeline, processes `dataset/messages.csv`, and writes predictions to `dataset/output.csv`.
- **`code/media_processor.py`**: Handles multimodal processing:
  - **OCR**: Uses PaddleOCR or Tesseract CLI (with OpenCV fallback for visual attachments).
  - **ASR**: Uses `faster-whisper` or `openai-whisper` to transcribe voice notes.
  - Caches extracted text to `dataset/media_cache.json` for fast re-runs.
- **`code/context_builder.py`**: Builds the relational context store across user preferences, group metadata, business account history, and past interaction events.
- **`code/retriever.py`**: Intelligent retrieval engine for finding historical evidence IDs (`evidence_message_ids`). Uses **TF-IDF Vectorization** with cosine similarity, full historical OCR/ASR text extraction, noise/stopword filtering, and user interaction event boosting (mutes, dismissals, reports, replies).
- **`code/router.py`**: Hybrid decision engine combining deterministic safety rules (prompt injection, phishing, domain spoofing, viral forwards), an optional **OpenAI `gpt-4o-mini` LLM pass** (when `OPENAI_API_KEY` is present), and an offline heuristic fallback pipeline for strict offline submission safety.

---

## Setup & Requirements

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) External CLI Dependencies

For OCR on images (if `paddleocr` is not used), install Tesseract OCR:
- **Windows**: Download installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

---

## How to Run

Execute the entry point from the repository root:

```bash
python code/main.py
```

This will:
1. Extract OCR/ASR features from `dataset/images/` and `dataset/audio/` (or load from `media_cache.json`).
2. Load historical context and relational data from `dataset/`.
3. Route each message in `dataset/messages.csv`.
4. Output the final predictions to `dataset/output.csv`.

---

## Output Contract

The generated `dataset/output.csv` follows the exact required schema:

| Column | Description |
|---|---|
| `message_id` | Unique ID of the incoming message |
| `action` | `notify`, `digest`, or `mute` |
| `message_type` | Category (e.g., `urgent`, `personal`, `event`, `promotion`, `scam`, `forward`, `greeting`, `business_update`) |
| `reason` | Human-readable explanation for the routing decision |
| `confidence` | Calibration score between `0.0` and `1.0` |
| `evidence_message_ids` | Comma-separated list of historical evidence IDs (or `none`) |
