"""
prepare_data.py — Downloads Quran data from CDN and builds FAISS vector index.
Run this script ONCE before launching the Streamlit app.

Data sources:
  Arabic  — quran-json CDN (original text)
  English — quran-json CDN (Saheeh International)
  Urdu    — fawazahmed0/quran-api (Maulana Fateh Muhammad Jalandhri, Deobandi)
"""

import os
import sys
import json
import pickle
import time
import requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Force UTF-8 output so emoji prints correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── CDN URLs ──────────────────────────────────────────────────────────────────
URLS = {
    "arabic":  "https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/quran.json",
    "english": "https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/quran_en.json",
    # Maulana Fateh Muhammad Jalandhri (Deobandi) via fawazahmed0/quran-api
    # Format differs from quran-json: root key "chapter", verses use "text" not "translation"
    "urdu":    "https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/editions/urd-fatehmuhammadja.json",
}

DATA_DIR   = Path("quran_data")
INDEX_DIR  = Path("faiss_index")
COMBINED_PATH = DATA_DIR / "quran_combined.json"
FAISS_PATH    = INDEX_DIR / "index.faiss"
META_PATH     = INDEX_DIR / "metadata.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE  = 256


# ── helpers ───────────────────────────────────────────────────────────────────

def fetch_json(url: str, retries: int = 3, delay: float = 2.0) -> dict | list:
    """Download JSON from *url* with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            print(f"   ↳ attempt {attempt}: GET {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"   ⚠️  Error: {exc}")
            if attempt < retries:
                print(f"   ⏳ Retrying in {delay}s …")
                time.sleep(delay)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts.")


# ── Step 1: Download ──────────────────────────────────────────────────────────

def download_data() -> tuple[list, dict, list]:
    """Download Arabic (list), Urdu (dict, fawazahmed0 format), and English (list)."""
    print("\n📥 Step 1: Downloading Quran data …")
    arabic_data  = fetch_json(URLS["arabic"])
    print("   ✅ Arabic JSON downloaded (quran-json format)")
    urdu_data    = fetch_json(URLS["urdu"])
    print("   ✅ Urdu JSON downloaded (fawazahmed0 format — Fateh Muhammad Jalandhri)")
    english_data = fetch_json(URLS["english"])
    print("   ✅ English JSON downloaded (quran-json format)")

    # Validate fawazahmed0 structure early for a clear error message
    # Actual structure: {"quran": [{"chapter": N, "verse": N, "text": "..."}, ...]}
    if not isinstance(urdu_data, dict) or "quran" not in urdu_data:
        raise RuntimeError(
            "Unexpected Urdu JSON structure from fawazahmed0 API. "
            f"Expected dict with 'quran' key, got keys: {list(urdu_data.keys()) if isinstance(urdu_data, dict) else type(urdu_data)}"
        )

    return arabic_data, urdu_data, english_data


# ── Step 2: Combine ───────────────────────────────────────────────────────────

def combine_data(
    arabic_data: list,
    urdu_data: dict,
    english_data: list,
) -> list[dict]:
    """Merge all three datasets into one list of verse records.

    Arabic + English use quran-json format (list of surahs, verses[].translation).
    Urdu uses fawazahmed0 format: {"quran": [{"chapter": N, "verse": N, "text": "..."}, ...]}
    — a flat list of all 6236 verses indexed by (chapter, verse) tuple.
    """
    print("\n🔀 Step 2: Combining data …")
    DATA_DIR.mkdir(exist_ok=True)

    # Build a (chapter, verse) → text lookup from the flat fawazahmed0 list
    urdu_lookup: dict[tuple[int, int], str] = {
        (v["chapter"], v["verse"]): v["text"]
        for v in urdu_data["quran"]
    }
    print(f"   ✅ Urdu lookup built: {len(urdu_lookup)} verse entries")

    combined: list[dict] = []

    for surah_idx, arabic_surah in enumerate(arabic_data):
        english_surah = english_data[surah_idx]

        surah_number       = arabic_surah.get("id", surah_idx + 1)
        surah_name_arabic  = arabic_surah.get("name", "")
        surah_name_english = arabic_surah.get("transliteration", "")

        arabic_verses  = arabic_surah.get("verses", [])
        english_verses = english_surah.get("verses", [])

        for ayah_idx, arabic_verse in enumerate(arabic_verses):
            ayah_number  = arabic_verse.get("id", ayah_idx + 1)
            arabic_text  = arabic_verse.get("text", "")

            english_text = ""
            if ayah_idx < len(english_verses):
                english_text = english_verses[ayah_idx].get("translation", "")

            # Look up Urdu text by (surah_number, ayah_number)
            urdu_text = urdu_lookup.get((surah_number, ayah_number), "")

            search_text = (
                f"Surah {surah_name_english} ({surah_number}), "
                f"Ayah {ayah_number}: {english_text} | {urdu_text}"
            )

            combined.append({
                "surah_number":       surah_number,
                "surah_name_arabic":  surah_name_arabic,
                "surah_name_english": surah_name_english,
                "ayah_number":        ayah_number,
                "arabic":             arabic_text,
                "urdu":               urdu_text,
                "english":            english_text,
                "search_text":        search_text,
            })

    with open(COMBINED_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"   ✅ {len(combined)} verses combined → {COMBINED_PATH}")
    return combined


# ── Step 3: Build FAISS index ─────────────────────────────────────────────────

def build_faiss_index(combined: list[dict]) -> None:
    """Encode search_text fields and save FAISS index + metadata."""
    import faiss  # local import so missing dep shows clear error

    print(f"\n🧠 Step 3: Building FAISS index (model: {EMBED_MODEL}) …")
    INDEX_DIR.mkdir(exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)
    print("   ✅ Embedding model loaded")

    texts = [v["search_text"] for v in combined]
    print(f"   ⏳ Encoding {len(texts)} verses (batch_size={BATCH_SIZE}) …")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    print("   ✅ Embeddings created:", embeddings.shape)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_PATH))
    print(f"   ✅ FAISS index saved → {FAISS_PATH}  ({index.ntotal} vectors)")

    with open(META_PATH, "wb") as f:
        pickle.dump(combined, f)
    print(f"   ✅ Metadata saved → {META_PATH}")

    print(f"\n🎉 Done! {index.ntotal} verses indexed and ready.")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Quran AI Assistant — Data Preparation Script")
    print("=" * 60)

    arabic_data, urdu_data, english_data = download_data()
    combined = combine_data(arabic_data, urdu_data, english_data)
    build_faiss_index(combined)

    print("\n✅ All done! You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
