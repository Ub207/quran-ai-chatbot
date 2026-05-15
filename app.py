"""
app.py — Quran AI Assistant: a RAG-based Streamlit chatbot.
Answers Quran questions in English or Urdu with Arabic text,
Urdu (Maulana Fateh Muhammad Jalandhri, Deobandi),
and English (Saheeh International) translations powered by Groq Llama 3.3 70B.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
FAISS_PATH = Path("faiss_index/index.faiss")
META_PATH  = Path("faiss_index/metadata.pkl")

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama-3.3-70b-versatile"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quran AI Assistant",
    page_icon="📖",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&display=swap');

    /* ── global ── */
    .stApp { background-color: #FAFFF7; }
    h1, h2, h3 { color: #1B5E20; }

    /* ── chat messages ── */
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }

    /* ── verse card ── */
    .verse-card {
        background: #F1F8E9;
        border-left: 5px solid #2E7D32;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    .verse-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #2E7D32;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }

    /* ── Arabic text ── */
    .arabic-text {
        font-family: 'Amiri', serif;
        font-size: 1.5rem;
        direction: rtl;
        text-align: right;
        line-height: 2.2;
        color: #1a1a1a;
        margin: 8px 0;
    }

    /* ── translation text ── */
    .translation-text {
        font-size: 0.95rem;
        color: #333;
        margin: 6px 0;
    }
    .urdu-text {
        font-size: 1.05rem;
        direction: rtl;
        text-align: right;
        line-height: 2.0;
        color: #333;
        margin: 6px 0;
    }

    /* ── sidebar ── */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #E8F5E9;
    }

    /* ── footer ── */
    .footer {
        text-align: center;
        padding: 20px 0 10px;
        color: #555;
        font-size: 0.85rem;
        border-top: 1px solid #C8E6C9;
        margin-top: 30px;
    }
    .footer .disclaimer {
        color: #777;
        font-size: 0.78rem;
        margin-top: 6px;
        font-style: italic;
    }

    /* ── example-question buttons ── */
    div[data-testid="stButton"] > button {
        background-color: #2E7D32;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.82rem;
        cursor: pointer;
        width: 100%;
        margin: 3px 0;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #1B5E20;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model …")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Loading Quran index …")
def load_faiss_index() -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "rb") as f:
        metadata: list[dict] = pickle.load(f)
    return index, metadata


@st.cache_resource(show_spinner="Connecting to Groq …")
def load_groq_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


# ── RAG helpers ───────────────────────────────────────────────────────────────

def search_quran(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """Return top_k most relevant verse records for *query*."""
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        rec = metadata[idx]
        results.append({
            "surah":         rec["surah_name_english"],
            "surah_arabic":  rec["surah_name_arabic"],
            "surah_number":  rec["surah_number"],
            "ayah":          rec["ayah_number"],
            "arabic":        rec["arabic"],
            "urdu":          rec["urdu"],
            "english":       rec["english"],
            "score":         float(dist),
        })
    return results


def build_context(results: list[dict]) -> str:
    """Format search results into an LLM-readable context block."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[Verse {i}] Surah {r['surah']} ({r['surah_number']}), Ayah {r['ayah']}\n"
            f"Arabic: {r['arabic']}\n"
            f"English: {r['english']}\n"
            f"Urdu: {r['urdu']}\n"
            f"{'-' * 60}"
        )
    return "\n".join(lines)


def get_ai_response(query: str, context: str, client: Groq) -> str:
    """Send query + context to Groq and return the assistant's reply."""
    system_prompt = (
        "You are a knowledgeable Quran AI Assistant created by Ubaid ur Rehman, "
        "an Aalim (Islamic Scholar) and AI Developer. "
        "Your role is to help users understand the Quran with accuracy and respect.\n\n"
        "The Urdu translation used is by Maulana Fateh Muhammad Jalandhri رحمه الله "
        "(Deobandi school), and the English translation is Saheeh International.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY from the provided Quranic context. Never fabricate verses.\n"
        "2. Always cite the Surah name and Ayah number for each verse you reference.\n"
        "3. Include the Arabic text, English translation, and Urdu translation in your answer.\n"
        "4. Be scholarly, respectful, and humble in tone.\n"
        "5. If the user writes in Urdu, respond in Urdu. Otherwise respond in English.\n"
        "6. If the context does not contain relevant information, say so honestly.\n"
        "7. Format Arabic verses clearly, indicating they are in Arabic script.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Relevant Quran verses for context:\n{context}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content


# ── Sidebar ───────────────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "What does the Quran say about patience?",
    "Sabr ke baare mein kya aaya hai?",
    "Tell me about Surah Al-Fatiha",
    "What are the verses about charity?",
    "Namaz ke baare mein ayaat",
    "What does the Quran say about parents?",
]

with st.sidebar:
    st.markdown("## 📖 Quran AI Assistant")
    st.markdown(
        "An AI-powered chatbot that answers your questions about the Holy Quran "
        "using semantic search and large language models.\n\n"
        "Ask in **English** or **Urdu** — I understand both."
    )
    st.divider()

    st.markdown("### 📚 Data Sources")
    st.markdown(
        "- **Arabic**: Original Quranic text\n"
        "- **English**: Saheeh International\n"
        "- **Urdu**: Maulana Fateh Muhammad Jalandhri رحمه الله"
    )
    st.divider()

    st.markdown("### 💡 Example Questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"btn_{q}"):
            st.session_state["pending_question"] = q

    st.divider()

    # Stats (only shown after index is loaded)
    if FAISS_PATH.exists():
        st.markdown("### 📊 Stats")
        st.markdown(f"- **Verses indexed:** 6,236\n- **Model:** `{LLM_MODEL}`")
        st.divider()

    st.markdown("### 🔗 Links")
    st.markdown(
        "- [GitHub](https://github.com/Ub207)\n"
        "- [Portfolio](https://github.com/Ub207)"
    )


# ── Guard: API key ────────────────────────────────────────────────────────────

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "**GROQ_API_KEY not found.**\n\n"
        "Please create a `.env` file with your key:\n"
        "```\nGROQ_API_KEY=your_key_here\n```\n\n"
        "Get a free key at [console.groq.com](https://console.groq.com)."
    )
    st.stop()

# ── Guard: FAISS index ────────────────────────────────────────────────────────

if not FAISS_PATH.exists() or not META_PATH.exists():
    st.error(
        "**FAISS index not found.**\n\n"
        "Please run the data preparation script first:\n"
        "```bash\npython prepare_data.py\n```\n\n"
        "This downloads the Quran data and builds the search index."
    )
    st.stop()

# ── Load resources ────────────────────────────────────────────────────────────

embed_model        = load_embedding_model()
faiss_index, meta  = load_faiss_index()
groq_client        = load_groq_client()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center; color:#1B5E20;'>📖 Quran AI Assistant</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#555; font-size:1.05rem;'>"
    "Ask any question about the Holy Quran in English or Urdu</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render existing chat history ──────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📜 View Source Verses", expanded=False):
                for v in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="verse-card">
                            <div class="label">Surah {v['surah']} ({v['surah_number']}), Ayah {v['ayah']}</div>
                            <div class="arabic-text">{v['arabic']}</div>
                            <div class="translation-text"><strong>English:</strong> {v['english']}</div>
                            <div class="urdu-text">{v['urdu']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ── Handle sidebar example-question button presses ────────────────────────────

if "pending_question" in st.session_state:
    pending = st.session_state.pop("pending_question")
    st.session_state["auto_query"] = pending
    st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────

auto_query = st.session_state.pop("auto_query", None)
user_input = st.chat_input("Ask about the Quran…") or auto_query

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching Quran & generating answer…"):
            try:
                results = search_quran(user_input, faiss_index, meta, embed_model)
                context = build_context(results)
                answer  = get_ai_response(user_input, context, groq_client)
            except Exception as exc:
                answer  = f"⚠️ An error occurred: {exc}\n\nPlease try again."
                results = []

        st.markdown(answer, unsafe_allow_html=True)

        if results:
            with st.expander("📜 View Source Verses", expanded=False):
                for v in results:
                    st.markdown(
                        f"""
                        <div class="verse-card">
                            <div class="label">Surah {v['surah']} ({v['surah_number']}), Ayah {v['ayah']}</div>
                            <div class="arabic-text">{v['arabic']}</div>
                            <div class="translation-text"><strong>English:</strong> {v['english']}</div>
                            <div class="urdu-text">{v['urdu']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": results,
    })

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="footer">
        Quran AI Assistant — Built with ❤️ by <strong>Ubaid ur Rehman</strong><br>
        <span>Aalim | Qari | AI Developer | Karachi, Pakistan</span><br>
        <span class="disclaimer">
            This is an AI tool for educational purposes.
            For authentic Islamic rulings, always consult qualified scholars.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
