# Movie Chatbot API

FastAPI backend for the Movie Chatbot RAG project, running directly (no Gradio UI used — this Space just hosts the raw API).

Note: `app.py` in this repo is a separate, standalone Streamlit version of the
chatbot, kept for local/alternative use — it is NOT the entry point for this
Space. This Space runs `space_entry.py` instead (see `app_file` above).

# 🎬 Movie Chatbot

A RAG-powered chatbot for exploring a movie plot dataset and your own uploaded documents — with per-user authentication, isolated file storage, a routing/verification pipeline built on LangGraph, and a multi-agent article generator (research → write → verify → human review → save).

Three frontends are available: a plain HTML page, a React app, and a Streamlit app — all talk to the same FastAPI backend.

---

## Features

- **JWT authentication** — signup/login, passwords hashed with bcrypt via passlib
- **Per-user file uploads** — each user's uploaded PDFs/DOCX/TXT files live in their own folder and are only ever searchable by that user; the base movie dataset stays shared and accessible to everyone
- **Isolated retrieval** — every chat query is filtered at the Qdrant database level by the current user's identity, so there's no code path where one user's question can retrieve another user's uploaded content
- **Smart chat routing (LangGraph)** — each question is classified as a comparison between two files, a question about your own files, a question about the movie dataset, or general knowledge — and answered accordingly, with the source always labeled in the reply
- **Answer verification** — a second pass checks that generated answers are actually supported by the retrieved context before returning them
- **Multi-agent article generator (LangGraph)** — give it a topic, and it researches (via Tavily web search), writes, and verifies an article, looping back to re-research up to 3 times if verification fails. You then review the draft, ask for edits to the title, body, or sources in plain language, and only save it once you confirm
- **Per-user article history** — every saved article is listed, reopenable, and deletable, scoped to its owner
- **Chat session history** — multiple named chat threads per user, same isolation guarantees as uploads

---

## Project structure

```
movie/
├── data/
│   ├── wiki_movie_plots_deduped.csv     # base movie dataset (shared, read-only)
│   ├── uploads/<user>/                  # per-user uploaded files
│   └── articles/<user>/                 # per-user saved articles (.md)
│
├── movie-chatbot-frontend/              # React frontend (Vite)
├── index.html                           # standalone HTML frontend
├── app.py                               # Streamlit frontend
│
├── server.py                            # FastAPI backend — all routes live here
├── auth.py                              # password hashing + JWT creation/verification
├── user_store.py                        # user accounts (JSON file)
├── chat_store.py                        # chat session history (JSON file)
├── rag_graph.py                         # LangGraph pipeline for /chat (routing, retrieval, verification)
├── article_pipeline.py                  # LangGraph pipeline for the article generator
├── build_database.py                    # builds/seeds the Qdrant movie dataset collection
├── cleanup_uploads.py                   # one-time script to wipe uploaded content from Qdrant
│
├── .env                                 # environment variables (not committed)
└── README.md
```

---

## Prerequisites

- Python 3.11+ and Node.js (for the React frontend)
- A [Qdrant](https://qdrant.tech) instance (cloud or self-hosted) with the movie dataset already loaded
- A Hugging Face API key (used for the chat LLM)
- A [Tavily](https://tavily.com) API key (used for the article generator's web search — free tier available, no credit card required)

---

## Setup

### 1. Clone and set up the virtual environment

```bash
git clone <your-repo-url>
cd movie

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux
```

### 2. Install backend dependencies

```bash
pip install -r requirements.txt
```

> If you don't have a `requirements.txt` yet, generate one from your active environment with `pip freeze > requirements.txt` before committing — this is what lets someone else (or future you) recreate this exact environment with one command instead of installing packages one at a time.

Key packages this project relies on: `fastapi`, `uvicorn`, `python-dotenv`, `pyjwt`, `passlib[bcrypt]`, `langchain-core`, `langchain-huggingface`, `langchain-qdrant`, `langchain-openai`, `langchain-classic`, `langchain-tavily`, `langgraph`, `qdrant-client`, `pypdf`, `python-docx`.

### 3. Configure environment variables

Create a `.env` file in the project root:

```
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
HF_API_KEY=your_huggingface_api_key
TAVILY_API_KEY=your_tavily_api_key
JWT_SECRET=a_long_random_secret_string
```

> ⚠️ `JWT_SECRET` must be set — don't rely on any fallback default. Anyone who knows the secret can forge valid login tokens, so this needs to be a real, private value, and `.env` should never be committed (make sure it's in `.gitignore`).

### 4. Build the movie dataset collection (first time only)

```bash
python build_database.py
```

This loads `wiki_movie_plots_deduped.csv` into your Qdrant collection.

---

## Running the backend

```bash
python -m uvicorn server:app --reload
```

The API runs at `http://127.0.0.1:8000`. Keep this running in its own terminal — all three frontends depend on it.

---

## Running a frontend

Pick one (or run more than one at a time, in separate terminals — they all just talk to the same backend):

### Option A — Plain HTML
No build step needed. With the backend running, just open `index.html` directly in your browser (or use a "Live Server" extension in VS Code).

### Option B — React (recommended, full feature set)
```bash
cd movie-chatbot-frontend
npm install
npm run dev
```
Open the local URL Vite prints (usually `http://localhost:5173`).

### Option C — Streamlit
```bash
streamlit run app.py
```

---

## One-time maintenance script: `cleanup_uploads.py`

If you need to wipe all previously uploaded content from Qdrant (e.g. resetting to a clean state with only the base dataset), run:

```bash
python cleanup_uploads.py
```

This deletes chunks tagged as uploads (leaving the base movie dataset untouched) and asks for confirmation before deleting anything. It's a one-time maintenance tool, not part of the running app — safe to delete after use, or keep it in the repo for future resets.

---

## How the article generator works

1. **Generate** — give it a topic; it searches the web (Tavily), writes a draft, and verifies the draft against its sources, retrying the research up to 3 times if verification fails. Nothing is saved yet.
2. **Revise** (optional, repeatable) — ask for changes in plain language (e.g. *"shorten the intro"*, *"change the title to X"*, *"remove the last source"*) — the whole letter (title, body, and sources) is editable this way, not just the body text.
3. **Confirm** — only this step actually writes the article to disk, under your own user folder.
4. **Discard** — throws away the draft with nothing saved.

Every saved article shows a **Verified** or **Unverified** badge, so you always know whether its content was confirmed against real sources.

---

## Security notes

- Passwords are hashed with bcrypt, never stored in plaintext
- JWTs expire after 7 days
- All file/article storage and retrieval is scoped per-user at the database query level, not just by folder structure
- `JWT_SECRET`, `QDRANT_API_KEY`, `HF_API_KEY`, and `TAVILY_API_KEY` should never be committed — confirm `.env` is listed in `.gitignore` before pushing