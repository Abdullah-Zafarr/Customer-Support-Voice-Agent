# 📚 Knowledge Base

Drop your documentation files here to power the Voice Agent's knowledge.

## Supported Formats
- `.txt` — Plain text files
- `.md` — Markdown files
- `.pdf` — PDF documents

## How It Works
1. Place your files in this folder.
2. Restart the server (or hit `POST /ingest` to re-index).
3. The agent will now answer questions using this documentation as context.

## Tips
- Shorter, focused documents work best.
- The system chunks documents into ~500 character segments for retrieval.
- You can add multiple files; the agent searches across all of them.
