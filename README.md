# Elevia

Session-based memory-aware chat assistant built with FastAPI and a fine-tuned LLaMA model.

## Features
- Session-scoped memory
- Public vs private facts
- Fact promotion via repetition
- Hallucination-resistant responses
- Per-user behavior profiles

## Run
```bash
pip install -r requirements.txt
uvicorn app.server:app --reload
