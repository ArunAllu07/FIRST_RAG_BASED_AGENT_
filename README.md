# Mini AI Agent

A command-line AI assistant that can answer questions using a Groq-hosted language model, search the web, multiply numbers, and retrieve relevant passages from a local PDF with ChromaDB.

## Features

- **PDF question answering:** retrieves relevant passages from `documents/SURFACE.pdf`.
- **Web search:** looks up recent or web-based information.
- **Calculator:** multiplies two numbers.
- **Conversation memory:** stores conversation checkpoints in a local SQLite database.

## Requirements

- Python 3.10 or later
- A Groq API key
- Internet access to call Groq, search the web, and download the embedding model on first use

## Setup

Run these commands from the project directory.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project directory and add your own Groq API key:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Keep `.env` private. Do not upload it or share your API key.

## Prepare the PDF database

The main agent searches the `surface_notes` collection under `chroma_db`. Build or refresh that collection from the included PDF:

```powershell
python store_vectors.py
```

The first run may download the `all-MiniLM-L6-v2` embedding model.

## Run the assistant

```powershell
python main.py
```

Ask questions at the `You:` prompt. Type `exit` to quit. For example:

```text
You: What does the PDF say about surface energy?
You: What is 12 multiplied by 8?
You: Search the web for recent information about ...
```

## Project files

| File or folder | Purpose |
| --- | --- |
| `main.py` | Runs the interactive assistant and registers the calculator, web search, and PDF search tools. |
| `store_vectors.py` | Extracts and chunks the PDF, creates embeddings, and stores them in `chroma_db/surface_notes`. |
| `retrieve.py` | Runs a standalone search against the `surface_notes` collection. |
| `create_embeddings.py` | Optional embedding workflow that writes to a separate `vector_store/surface_pdf` collection; `main.py` does not use this collection. |
| `chunk_pdf.py`, `test_pdf.py` | PDF text extraction and chunking utilities. |
| `documents/SURFACE.pdf` | Source PDF used by the indexing scripts. |
| `agent_memory.db` | Local SQLite conversation checkpoint database created by the agent. |
| `chroma_db/`, `vector_store/` | Local vector databases created by the indexing scripts. |

Run the scripts from the project directory so their relative file paths resolve correctly.

## Uploading to GitHub

The included `.gitignore` excludes `.env`, virtual environments, SQLite memory files, and generated vector databases. Review your staged files before pushing and make sure no API keys or other private data are included.
