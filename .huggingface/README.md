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

A RAG-based chatbot that answers questions about the Holy Quran in **English** and **Urdu**, with Arabic text, English (Saheeh International), and Urdu (Maulana Fateh Muhammad Jalandhri رحمه الله) translations.

Built with:
- Streamlit
- FAISS vector search over all 6,236 verses
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Groq LLM API (with primary + fallback model support)

## How to deploy this Space

1. **Fork or clone** this repo to your Hugging Face account.
2. The Space auto-detects `sdk: streamlit` from the README header.
3. Go to **Settings → Repository secrets** and add:
   - **Name:** `GROQ_API_KEY`
   - **Value:** your key from [console.groq.com/keys](https://console.groq.com/keys)
4. Optional model overrides (also in Repository secrets):
   - `GROQ_MODEL_PRIMARY`  (default: `openai/gpt-oss-20b`, free tier)
   - `GROQ_MODEL_FALLBACK` (default: `openai/gpt-oss-120b`, free tier)
5. The Space will rebuild and serve the app on port 8501.

## Local development

```bash
pip install -r requirements.txt
cp .env.example .env  # then edit with your GROQ_API_KEY
python prepare_data.py        # one-time: builds the FAISS index
python verify.py              # sanity check before launching
streamlit run app.py
```

## Data sources

- Arabic text & English (Saheeh International): [quran-json](https://github.com/semarketir/quranjson)
- Urdu (Maulana Fateh Muhammad Jalandhri رحمه الله): [fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api)

## Credits

Built by **Ubaid ur Rehman** — Aalim | Qari | AI Developer | Karachi, Pakistan.

MIT License.
