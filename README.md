---
title: Quran AI Assistant
emoji: 📖
colorFrom: green
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: RAG chatbot for the Quran in English & Urdu powered by Groq
---

# 📖 Quran AI Assistant

> A production-ready RAG-based chatbot that answers questions about the Holy Quran in English and Urdu, powered by semantic search and Groq's Llama models.

---

## ✨ Features

- **Semantic Search** — FAISS vector search across all 6,236 Quranic verses
- **Trilingual** — Arabic text, Urdu translation (Maulana Fateh Muhammad Jalandhri رحمه الله), English translation (Saheeh International)
- **RAG Architecture** — Retrieval-Augmented Generation for grounded, citation-backed answers
- **Bilingual Chat** — Ask in English or Urdu, get answers in the same language
- **Model Fallback** — Primary + secondary Groq model with automatic failover
- **Startup Validation** — Connectivity check before serving users
- **Islamic UI** — Green Islamic theme, Amiri font for Arabic, right-to-left Urdu support
- **Free & Deployable** — Runs on Hugging Face Spaces, Streamlit Cloud, or locally

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq API — `openai/gpt-oss-20b` (primary) / `openai/gpt-oss-120b` (fallback) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS-cpu |
| Data (Arabic + English) | quran-json CDN |
| Data (Urdu) | fawazahmed0/quran-api (Fateh Muhammad Jalandhri) |
| Env | python-dotenv |

> **Note:** All Meta Llama models on Groq are now **Enterprise tier (paid only)**. The app defaults to the `openai/gpt-oss-*` family, which is the current free-tier option on the Groq dev plan. If you have an Enterprise plan, you can override `GROQ_MODEL_PRIMARY` with `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`.

---

## 🏗 Architecture

```
User Question
      │
      ▼
Sentence Transformer (all-MiniLM-L6-v2)
      │  encodes query into 384-dim vector
      ▼
FAISS IndexFlatL2
      │  returns top-5 nearest verse embeddings
      ▼
Top 5 Verses (Arabic + English + Urdu)
      │  formatted as context
      ▼
Groq LLM  (primary → fallback → free-tier list if needed)
      │  generates scholarly answer
      ▼
Streamlit Chat Response
(Arabic + English + Urdu + Citations)
```

---

## 🚀 Quick Start (Local)

**1. Clone the repo**
```bash
git clone https://github.com/Ub207/quran-ai-chatbot
cd quran-ai-chatbot
```

**2. Create & activate a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your Groq API key**
```bash
cp .env.example .env
# Edit .env and add your key:  GROQ_API_KEY=gsk_...
# Get a free key at: https://console.groq.com
```

**5. Prepare data (run once)**
```bash
python prepare_data.py
```
This downloads the Quran JSON files and builds the FAISS index (~2 minutes).

**6. Verify before launching**
```bash
python verify.py
```
Confirms the API key works, both models are reachable, and the FAISS index is valid.

**7. Launch the app**
```bash
streamlit run app.py
```

---

## ⚙️ Configuration

All configuration is via environment variables (or HF Spaces / Streamlit Cloud secrets):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL_PRIMARY` | `openai/gpt-oss-20b` | First-choice model |
| `GROQ_MODEL_FALLBACK` | `openai/gpt-oss-120b` | Used if primary returns 404 |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model |
| `LOG_LEVEL` | `INFO` | `DEBUG` for verbose request logs |

The app automatically **falls back** to `GROQ_MODEL_FALLBACK` if the primary model is unavailable (404, rate limit, etc.).

---

## 🤗 Deploy on Hugging Face Spaces

The repo includes the `sdk: streamlit` header at the top of this README, so HF Spaces will auto-detect it as a Streamlit app.

1. **Create a new Space** at [huggingface.co/new-space](https://huggingface.co/new-space):
   - **SDK:** Streamlit
   - **Hardware:** CPU basic (free) is enough — `sentence-transformers` runs in-memory
   - **Visibility:** Public or Private
2. **Push this repo** to the Space's git remote (HF gives you the URL on the Space page):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/quran-ai-assistant
   git push space main
   ```
3. **Add the `GROQ_API_KEY` secret**:
   - Space page → **Settings** → **Repository secrets** → **New secret**
   - **Name:** `GROQ_API_KEY`
   - **Value:** your key from [console.groq.com/keys](https://console.groq.com/keys)
4. Optional: add `GROQ_MODEL_PRIMARY` and `GROQ_MODEL_FALLBACK` secrets to override the defaults. The hardcoded free-tier list inside `app.py` (in `FREE_TIER_FALLBACKS`) is the last-resort set that's probed if both fail.
5. The Space builds (~2-3 min for the first run because of `torch` + `sentence-transformers`) and serves on port 8501.

### Hugging Face Spaces deployment checklist

- [ ] README has YAML frontmatter starting with `sdk: streamlit` (✅ included)
- [ ] `app.py` is at the repo root (✅)
- [ ] `requirements.txt` is at the repo root (✅)
- [ ] `runtime.txt` pins Python 3.11.9 (✅)
- [ ] `faiss_index/index.faiss` and `faiss_index/metadata.pkl` are committed
- [ ] `quran_data/quran_combined.json` is committed (optional, only if you want to re-run `prepare_data.py`)
- [ ] `GROQ_API_KEY` added under **Settings → Repository secrets**

> **Note on `packages.txt`:** This file is a Streamlit Cloud convention. HF Spaces ignores it. The `libgomp1` system lib that `sentence-transformers` needs is already included in HF's default Space image.

---

## ☁️ Deploy on Streamlit Cloud

1. **Push to GitHub** with the `faiss_index/` and `quran_data/` folders committed.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select this repo.
3. Add the following in **Settings → Secrets**:
   ```toml
   GROQ_API_KEY         = "gsk_..."
   GROQ_MODEL_PRIMARY   = "openai/gpt-oss-20b"
   GROQ_MODEL_FALLBACK  = "openai/gpt-oss-120b"
   ```
4. Click **Deploy**. The app runs `verify.py`-equivalent checks at startup before serving users.

---

### Deployment Checklist (works for both HF Spaces & Streamlit Cloud)

- [ ] `GROQ_API_KEY` set in your platform's secrets
- [ ] `GROQ_MODEL_PRIMARY` set (default: `openai/gpt-oss-20b`)
- [ ] `GROQ_MODEL_FALLBACK` set (default: `openai/gpt-oss-120b`)
- [ ] `faiss_index/index.faiss` and `faiss_index/metadata.pkl` present in repo
- [ ] `quran_data/quran_combined.json` present in repo
- [ ] `requirements.txt` pinned versions
- [ ] Local `python verify.py` returns ✅ for all checks

---

## 🐛 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GROQ_API_KEY not found` | Key not in env or secrets | Add to `.env`, Streamlit Cloud secrets, or HF Spaces **Repository secrets** |
| `model does not exist or you do not have access` (404) | Groq renamed the model, or your account tier doesn't have access to it | The app now defaults to `openai/gpt-oss-20b` (the current free-tier option on the Groq dev plan). Set `GROQ_MODEL_PRIMARY` to override. |
| `FAISS index not found` | Index not committed | Run `python prepare_data.py` locally and commit `faiss_index/` |
| HF Space stuck on "Building" | Missing `sdk: streamlit` in README header | Already added — re-check by viewing the raw README on the Space |
| HF Space `ModuleNotFoundError` at boot | Pinning too tight, or `requirements.txt` missing | Confirm `requirements.txt` is at repo root and `runtime.txt` pins Python 3.11.9 |
| Slow first response | Cold-start loading embedding model | Normal — cached after first run |
| Rate limit errors | Free-tier quota exceeded | App will auto-fallback to the secondary model |

---

## 📂 Project Structure

```
quran-ai-chatbot/
├── app.py                  # Streamlit chatbot app
├── prepare_data.py         # Data download + FAISS index builder
├── verify.py               # Pre-deployment verification
├── requirements.txt        # Python dependencies (pinned)
├── runtime.txt             # Python version pin (HF Spaces)
├── .env.example            # Environment variable template
├── .gitignore
├── README.md               # Includes HF Spaces YAML frontmatter
├── .huggingface/
│   └── README.md           # Space metadata (description, tags)
├── .streamlit/
│   └── config.toml         # Streamlit config
├── quran_data/             # auto-generated
│   └── quran_combined.json
└── faiss_index/            # auto-generated
    ├── index.faiss
    └── metadata.pkl
```

---

## 🙏 Credits

Built by **Ubaid ur Rehman** — Aalim | Qari | AI Developer | Karachi, Pakistan

- [GitHub](https://github.com/Ub207)

**Data credits:**
- Arabic text & English (Saheeh International): [quran-json](https://github.com/semarketir/quranjson)
- Urdu (Maulana Fateh Muhammad Jalandhri رحمه الله): [fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api)

---

## 📄 License

MIT License — free to use, modify, and distribute.
