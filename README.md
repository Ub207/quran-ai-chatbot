# 📖 Quran AI Assistant

> A production-ready RAG-based chatbot that answers questions about the Holy Quran in English and Urdu, powered by semantic search and Groq's Llama 3.3 70B.

---

## ✨ Features

- **Semantic Search** — FAISS vector search across all 6,236 Quranic verses
- **Trilingual** — Arabic text, Urdu translation (Maulana Fateh Muhammad Jalandhri رحمه الله), English translation (Saheeh International)
- **RAG Architecture** — Retrieval-Augmented Generation for grounded, citation-backed answers
- **Bilingual Chat** — Ask in English or Urdu, get answers in the same language
- **Islamic UI** — Green Islamic theme, Amiri font for Arabic, right-to-left Urdu support
- **Free & Deployable** — Runs on Streamlit Cloud with Groq's free tier

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS-cpu |
| Data (Arabic + English) | quran-json CDN |
| Data (Urdu) | fawazahmed0/quran-api (Fateh Muhammad Jalandhri) |
| Env | python-dotenv |

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
Groq LLM (llama-3.3-70b-versatile)
      │  generates scholarly answer
      ▼
Streamlit Chat Response
(Arabic + English + Urdu + Citations)
```

---

## 🚀 Quick Start

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

**6. Launch the app**
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub (the `faiss_index/` and `quran_data/` folders are gitignored — generate them locally first, then either commit them or run `prepare_data.py` as part of a setup script).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo.
3. Add `GROQ_API_KEY` in **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
4. For production deployments, commit the generated `faiss_index/` and `quran_data/` folders (remove them from `.gitignore`) so Streamlit Cloud does not need to rebuild the index on every cold start.

---

## 📂 Project Structure

```
quran-ai-chatbot/
├── app.py                  # Streamlit chatbot app
├── prepare_data.py         # Data download + FAISS index builder
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── README.md
├── quran_data/             # auto-generated
│   └── quran_combined.json
└── faiss_index/            # auto-generated
    ├── index.faiss
    └── metadata.pkl
```

---

## 🖼 Screenshots

> *(Coming soon)*

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
