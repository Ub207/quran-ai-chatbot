"""
app.py — Quran AI Assistant: a RAG-based Streamlit chatbot.

Answers Quran questions in English or Urdu with Arabic text,
Urdu (Maulana Fateh Muhammad Jalandhri, Deobandi),
and English (Saheeh International) translations powered by Groq.

Model configuration (override via environment variables):
  GROQ_MODEL_PRIMARY   — primary model  (default: openai/gpt-oss-20b, free tier)
  GROQ_MODEL_FALLBACK  — fallback model (default: openai/gpt-oss-120b, free tier)
  GROQ_API_KEY         — required Groq API key

Notes on model selection (2026-09):
  All Meta Llama models on Groq are now "Enterprise" tier (paid accounts only).
  For free-tier accounts, use the openai/gpt-oss-* models, which are
  open-weight and have the same chat-completions interface.
"""

import logging
import os
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AuthenticationError,
    Groq,
    NotFoundError,
    RateLimitError,
)
from sentence_transformers import SentenceTransformer

# ── Version & startup banner ──────────────────────────────────────────────────
#
# Bump APP_VERSION every time you change app.py. This makes it trivial to
# verify in the running container that the latest code is actually deployed:
#   - It appears in the Streamlit UI (homepage caption)
#   - It appears in the first 2 lines of every startup log
APP_VERSION = "2026-09-07-surah-summary-v1"
print(f"APP VERSION: {APP_VERSION}")
print(f"Python: {os.sys.version.split()[0]}  Streamlit: {st.__version__}")

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format=LOG_FORMAT)
logger = logging.getLogger("quran-ai")
logger.info("=" * 70)
logger.info("APP VERSION: %s", APP_VERSION)
logger.info("=" * 70)

# Load .env from the script directory (not the working directory)
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)


# ── Configuration ────────────────────────────────────────────────────────────

# Defaults reflect the *current* Groq free-tier production line as of 2026-09.
# All Meta Llama models on Groq are now Enterprise tier (paid only).
# Override via env vars or Streamlit / HF Spaces secrets.
DEFAULT_PRIMARY_MODEL  = "openai/gpt-oss-20b"     # free tier, 1000 t/s
DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-120b"    # free tier, 500 t/s
DEFAULT_EMBED_MODEL    = "all-MiniLM-L6-v2"


def _get_secret(name: str, default: str = "") -> str:
    """Read from environment first (HF Spaces & local), then Streamlit secrets, then default."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        return st.secrets.get(name, default)  # type: ignore[attr-defined]
    except Exception:
        return default


def _get_api_key() -> str:
    """Return the Groq API key from env or Streamlit / HF Spaces secrets."""
    return _get_secret("GROQ_API_KEY", "")


def _get_model(name: str, default: str) -> str:
    return _get_secret(name, default) or default


GROQ_MODEL_PRIMARY  = _get_model("GROQ_MODEL_PRIMARY",  DEFAULT_PRIMARY_MODEL)
GROQ_MODEL_FALLBACK = _get_model("GROQ_MODEL_FALLBACK", DEFAULT_FALLBACK_MODEL)
EMBED_MODEL         = _get_model("EMBED_MODEL",        DEFAULT_EMBED_MODEL)


# ── Paths ─────────────────────────────────────────────────────────────────────
FAISS_PATH = Path("faiss_index/index.faiss")
META_PATH  = Path("faiss_index/metadata.pkl")


# ── Surah database ────────────────────────────────────────────────────────────
#
# Comprehensive list of all 114 surahs. For each, we store:
#   number        — official 1-114
#   english       — canonical English name (lowercase, no "Surah" prefix)
#   arabic        — Arabic name
#   transliteration — common romanized spellings (multiple variants)
#   urdu_script   — Urdu-script name if it differs
#
# This is the single source of truth for Surah name detection.

SURAH_DB: list[dict] = [
    {"number": 1,   "english": "fatihah",     "arabic": "الفاتحة",   "transliteration": ["fatihah", "fatiha", "fatih", "faatihah", "faatiha"], "urdu_script": "الفاتحہ"},
    {"number": 2,   "english": "baqarah",     "arabic": "البقرة",    "transliteration": ["baqarah", "baqara", "baqar", "baqqarah", "baqrah"], "urdu_script": "البقرہ"},
    {"number": 3,   "english": "imran",       "arabic": "آل عمران",  "transliteration": ["imran", "aali imran", "ali imran", "aal imran", "al imran", "aale imran"], "urdu_script": "آل عمران"},
    {"number": 4,   "english": "nisa",        "arabic": "النساء",    "transliteration": ["nisa", "nisaa", "nisa'", "an nisa"], "urdu_script": "النساء"},
    {"number": 5,   "english": "maidah",      "arabic": "المائدة",   "transliteration": ["maidah", "maida", "maedah", "al-maidah", "al maidah", "al maida", "al maedah", "al-maida", "al-maedah"], "urdu_script": "المائدہ"},
    {"number": 6,   "english": "anam",        "arabic": "الأنعام",   "transliteration": ["anam", "an'am", "an am", "an aam", "ala'nam", "al-anam", "an'am"], "urdu_script": "الانعام"},
    {"number": 7,   "english": "araf",        "arabic": "الأعراف",   "transliteration": ["araf", "a'raf", "ar'af", "al-araf", "al araf", "al a'raf"], "urdu_script": "الاعراف"},
    {"number": 8,   "english": "anfal",       "arabic": "الأنفال",   "transliteration": ["anfal", "anfal", "al-anfal", "anfal"], "urdu_script": "الانفال"},
    {"number": 9,   "english": "tawbah",      "arabic": "التوبة",    "transliteration": ["tawbah", "tawba", "taubah", "tauba", "tubah", "touba", "tawbah", "at-tawbah", "al-tawbah", "tubah", "toba", "tawba"], "urdu_script": "التوبہ"},
    {"number": 10,  "english": "yunus",       "arabic": "يونس",      "transliteration": ["yunus", "younus", "yunus", "yunas"], "urdu_script": "یونس"},
    {"number": 11,  "english": "hud",         "arabic": "هود",       "transliteration": ["hud", "hood"], "urdu_script": "ہود"},
    {"number": 12,  "english": "yusuf",       "arabic": "يوسف",      "transliteration": ["yusuf", "yousuf", "yoosuf", "yusof", "yosuf"], "urdu_script": "یوسف"},
    {"number": 13,  "english": "rad",         "arabic": "الرعد",     "transliteration": ["rad", "ra'd", "ar-rad", "al-rad"], "urdu_script": "الرعد"},
    {"number": 14,  "english": "ibrahim",     "arabic": "إبراهيم",   "transliteration": ["ibrahim", "ibraheem", "ibrahem"], "urdu_script": "ابراہیم"},
    {"number": 15,  "english": "hijr",        "arabic": "الحجر",     "transliteration": ["hijr", "al-hijr", "al hijr"], "urdu_script": "الحجر"},
    {"number": 16,  "english": "nahl",        "arabic": "النحل",     "transliteration": ["nahl", "an-nahl", "al nahl"], "urdu_script": "النحل"},
    {"number": 17,  "english": "isra",        "arabic": "الإسراء",   "transliteration": ["isra", "israa", "bani israel", "isra'", "al-isra"], "urdu_script": "بنی اسرائیل"},
    {"number": 18,  "english": "kahf",        "arabic": "الكهف",     "transliteration": ["kahf", "al-kahf", "al kahf", "kaaf", "kahaf"], "urdu_script": "کہف"},
    {"number": 19,  "english": "maryam",      "arabic": "مريم",      "transliteration": ["maryam", "mariam", "maryum"], "urdu_script": "مریم"},
    {"number": 20,  "english": "taha",        "arabic": "طه",        "transliteration": ["taha", "ta ha", "taaha"], "urdu_script": "طٰہٰ"},
    {"number": 21,  "english": "anbiya",      "arabic": "الأنبياء",  "transliteration": ["anbiya", "ambiya", "anbiya'", "al-anbiya"], "urdu_script": "الانبیاء"},
    {"number": 22,  "english": "hajj",        "arabic": "الحج",      "transliteration": ["hajj", "haj"], "urdu_script": "حج"},
    {"number": 23,  "english": "muminun",     "arabic": "المؤمنون",  "transliteration": ["muminun", "mominun", "muminoon", "al-muminun", "al muminun"], "urdu_script": "المومنون"},
    {"number": 24,  "english": "nur",         "arabic": "النور",     "transliteration": ["nur", "noor", "al-nur", "al nur"], "urdu_script": "النور"},
    {"number": 25,  "english": "furqan",      "arabic": "الفرقان",   "transliteration": ["furqan", "al-furqan", "al furqan", "furkaan"], "urdu_script": "فرقان"},
    {"number": 26,  "english": "shuara",      "arabic": "الشعراء",   "transliteration": ["shuara", "shu'ara", "al-shuara", "al shuara"], "urdu_script": "الشعراء"},
    {"number": 27,  "english": "naml",        "arabic": "النمل",     "transliteration": ["naml", "an-naml", "al naml", "al-naml", "namal", "namul"], "urdu_script": "النمل"},
    {"number": 28,  "english": "qasas",       "arabic": "القصص",     "transliteration": ["qasas", "qasas", "al-qasas", "qisaas", "qasas", "qissaas"], "urdu_script": "القصص"},
    {"number": 29,  "english": "ankabut",     "arabic": "العنكبوت",  "transliteration": ["ankabut", "ankaboot", "al-ankabut", "anqabut"], "urdu_script": "العنکبوت"},
    {"number": 30,  "english": "rum",         "arabic": "الروم",     "transliteration": ["rum", "roam", "al-rum", "ar-rum"], "urdu_script": "الروم"},
    {"number": 31,  "english": "luqman",      "arabic": "لقمان",     "transliteration": ["luqman", "luqmaan", "loqman"], "urdu_script": "لقمان"},
    {"number": 32,  "english": "sajdah",      "arabic": "السجدة",    "transliteration": ["sajdah", "sajda", "al-sajdah", "sijdah"], "urdu_script": "سجدہ"},
    {"number": 33,  "english": "ahzab",       "arabic": "الأحزاب",   "transliteration": ["ahzab", "ahzaab", "al-ahzab"], "urdu_script": "احزاب"},
    {"number": 34,  "english": "saba",        "arabic": "سبأ",       "transliteration": ["saba", "saba'", "al-saba"], "urdu_script": "سبا"},
    {"number": 35,  "english": "fatir",       "arabic": "فاطر",      "transliteration": ["fatir", "faatir", "al-fatir"], "urdu_script": "فاطر"},
    {"number": 36,  "english": "ya sin",      "arabic": "يس",        "transliteration": ["ya sin", "yaseen", "yaaseen", "ya-seen", "yaseen"], "urdu_script": "یٰس"},
    {"number": 37,  "english": "saffat",      "arabic": "الصافات",   "transliteration": ["saffat", "saaffaat", "al-saffat"], "urdu_script": "صافات"},
    {"number": 38,  "english": "sad",         "arabic": "ص",         "transliteration": ["sad", "saad", "suad"], "urdu_script": "صٓ"},
    {"number": 39,  "english": "zumar",       "arabic": "الزمر",     "transliteration": ["zumar", "az-zumar", "al zumar", "zumar"], "urdu_script": "زمر"},
    {"number": 40,  "english": "ghafir",      "arabic": "غافر",      "transliteration": ["ghafir", "ghaafir", "al-ghafir", "momin"], "urdu_script": "غافر"},
    {"number": 41,  "english": "fusilat",     "arabic": "فصلت",     "transliteration": ["fusilat", "fussilat", "fussilaat"], "urdu_script": "فصلت"},
    {"number": 42,  "english": "shura",       "arabic": "الشورى",   "transliteration": ["shura", "shuura", "shuraa", "al-shura"], "urdu_script": "شوریٰ"},
    {"number": 43,  "english": "zukhruf",     "arabic": "الزخرف",   "transliteration": ["zukhruf", "zukhruf", "az-zukhruf"], "urdu_script": "زخرف"},
    {"number": 44,  "english": "dukhan",      "arabic": "الدخان",   "transliteration": ["dukhan", "dukhon", "ad-dukhan"], "urdu_script": "دخان"},
    {"number": 45,  "english": "jathiyah",    "arabic": "الجاثية",  "transliteration": ["jathiyah", "jaasiyah", "al-jathiyah"], "urdu_script": "جاثیہ"},
    {"number": 46,  "english": "ahqaf",       "arabic": "الأحقاف",  "transliteration": ["ahqaf", "ahqaaf", "al-ahqaf"], "urdu_script": "احقاف"},
    {"number": 47,  "english": "muhammad",    "arabic": "محمد",      "transliteration": ["muhammad", "mohammad", "muhamad"], "urdu_script": "محمد"},
    {"number": 48,  "english": "fath",        "arabic": "الفتح",    "transliteration": ["fath", "fatah", "al-fath", "fath"], "urdu_script": "فتح"},
    {"number": 49,  "english": "hujurat",     "arabic": "الحجرات",  "transliteration": ["hujurat", "hujuraat", "al-hujurat"], "urdu_script": "حجرات"},
    {"number": 50,  "english": "qaf",         "arabic": "ق",        "transliteration": ["qaf", "qaaf"], "urdu_script": "قٓ"},
    {"number": 51,  "english": "dhariyat",    "arabic": "الذاريات", "transliteration": ["dhariyat", "dhaariyaat", "ad-dhariyat"], "urdu_script": "ذاریات"},
    {"number": 52,  "english": "tur",         "arabic": "الطور",    "transliteration": ["tur", "al-tur", "ttoor"], "urdu_script": "طور"},
    {"number": 53,  "english": "najm",        "arabic": "النجم",    "transliteration": ["najm", "najam", "an-najm"], "urdu_script": "نجم"},
    {"number": 54,  "english": "qamar",       "arabic": "القمر",    "transliteration": ["qamar", "al-qamar"], "urdu_script": "قمر"},
    {"number": 55,  "english": "rahman",      "arabic": "الرحمن",   "transliteration": ["rahman", "al-rahman", "ar-rahman", "rehman"], "urdu_script": "رحمان"},
    {"number": 56,  "english": "waqiah",      "arabic": "الواقعة",  "transliteration": ["waqiah", "waaqiah", "al-waqiah"], "urdu_script": "واقعہ"},
    {"number": 57,  "english": "hadid",       "arabic": "الحديد",   "transliteration": ["hadid", "al-hadid"], "urdu_script": "حدید"},
    {"number": 58,  "english": "mujadilah",   "arabic": "المجادلة", "transliteration": ["mujadilah", "mujaadilah", "al-mujadilah"], "urdu_script": "مجادلہ"},
    {"number": 59,  "english": "hashr",       "arabic": "الحشر",    "transliteration": ["hashr", "al-hashr"], "urdu_script": "حشر"},
    {"number": 60,  "english": "mumtahanah",  "arabic": "الممتحنة", "transliteration": ["mumtahanah", "mumtahana", "al-mumtahanah"], "urdu_script": "ممتحنہ"},
    {"number": 61,  "english": "saff",        "arabic": "الصف",     "transliteration": ["saff", "as-saff"], "urdu_script": "صف"},
    {"number": 62,  "english": "jumuah",      "arabic": "الجمعة",   "transliteration": ["jumuah", "juma", "jum'ah", "al-jumuah"], "urdu_script": "جمعہ"},
    {"number": 63,  "english": "munafiqun",   "arabic": "المنافقون","transliteration": ["munafiqun", "munaafiqoon", "al-munafiqun"], "urdu_script": "منافقون"},
    {"number": 64,  "english": "taghabun",    "arabic": "التغابن",  "transliteration": ["taghabun", "at-taghabun", "tghabun"], "urdu_script": "تغابن"},
    {"number": 65,  "english": "talaq",       "arabic": "الطلاق",   "transliteration": ["talaq", "at-talaq"], "urdu_script": "طلاق"},
    {"number": 66,  "english": "tahrim",      "arabic": "التحريم",  "transliteration": ["tahrim", "at-tahrim"], "urdu_script": "تحریم"},
    {"number": 67,  "english": "mulk",        "arabic": "الملك",    "transliteration": ["mulk", "al-mulk"], "urdu_script": "ملک"},
    {"number": 68,  "english": "qalam",       "arabic": "القلم",    "transliteration": ["qalam", "al-qalam", "nuun"], "urdu_script": "قلم"},
    {"number": 69,  "english": "haqqah",      "arabic": "الحاقة",   "transliteration": ["haqqah", "al-haqqah", "haqah"], "urdu_script": "حاقہ"},
    {"number": 70,  "english": "maarij",      "arabic": "المعارج",  "transliteration": ["maarij", "maaarij", "al-maarij"], "urdu_script": "معارج"},
    {"number": 71,  "english": "nuh",         "arabic": "نوح",      "transliteration": ["nuh", "nooh"], "urdu_script": "نوح"},
    {"number": 72,  "english": "jinn",        "arabic": "الجن",     "transliteration": ["jinn", "jin", "al-jinn"], "urdu_script": "جن"},
    {"number": 73,  "english": "muzzammil",   "arabic": "المزمل",   "transliteration": ["muzzammil", "muzammil", "al-muzzammil"], "urdu_script": "مزمل"},
    {"number": 74,  "english": "muddaththir", "arabic": "المدثر",   "transliteration": ["muddaththir", "mudathir", "al-muddaththir"], "urdu_script": "مدثر"},
    {"number": 75,  "english": "qiyamah",     "arabic": "القيامة",  "transliteration": ["qiyamah", "qiyama", "al-qiyamah"], "urdu_script": "قیامہ"},
    {"number": 76,  "english": "insan",       "arabic": "الإنسان",  "transliteration": ["insan", "al-insan", "dahr"], "urdu_script": "انسان"},
    {"number": 77,  "english": "mursalat",    "arabic": "المرسلات", "transliteration": ["mursalat", "mursalaat", "al-mursalat"], "urdu_script": "مرسلات"},
    {"number": 78,  "english": "naba",        "arabic": "النبأ",    "transliteration": ["naba", "an-naba"], "urdu_script": "نباء"},
    {"number": 79,  "english": "naziat",      "arabic": "النازعات", "transliteration": ["naziat", "naazi'aat", "an-naziat"], "urdu_script": "نازعات"},
    {"number": 80,  "english": "abasa",       "arabic": "عبس",      "transliteration": ["abasa", "al-abasa"], "urdu_script": "عبس"},
    {"number": 81,  "english": "takwir",      "arabic": "التكوير",  "transliteration": ["takwir", "at-takwir"], "urdu_script": "تکویر"},
    {"number": 82,  "english": "infitar",     "arabic": "الإنفطار", "transliteration": ["infitar", "al-infitar"], "urdu_script": "انفطار"},
    {"number": 83,  "english": "mutaffifin",  "arabic": "المطففين", "transliteration": ["mutaffifin", "mutaffifeen", "al-mutaffifin"], "urdu_script": "مطففین"},
    {"number": 84,  "english": "inshiqaq",    "arabic": "الإنشقاق", "transliteration": ["inshiqaq", "al-inshiqaq"], "urdu_script": "انشقاق"},
    {"number": 85,  "english": "buruj",       "arabic": "البروج",   "transliteration": ["buruj", "al-buruj"], "urdu_script": "بروج"},
    {"number": 86,  "english": "tariq",       "arabic": "الطارق",   "transliteration": ["tariq", "at-tariq"], "urdu_script": "طارق"},
    {"number": 87,  "english": "ala",         "arabic": "الأعلى",   "transliteration": ["ala", "a'la", "al-a'la"], "urdu_script": "اعلیٰ"},
    {"number": 88,  "english": "ghashiyah",   "arabic": "الغاشية", "transliteration": ["ghashiyah", "al-ghashiyah"], "urdu_script": "غاشیہ"},
    {"number": 89,  "english": "fajr",        "arabic": "الفجر",    "transliteration": ["fajr", "al-fajr"], "urdu_script": "فجر"},
    {"number": 90,  "english": "balad",       "arabic": "البلد",    "transliteration": ["balad", "al-balad"], "urdu_script": "بلد"},
    {"number": 91,  "english": "shams",       "arabic": "الشمس",    "transliteration": ["shams", "ash-shams"], "urdu_script": "شمس"},
    {"number": 92,  "english": "layl",        "arabic": "الليل",    "transliteration": ["layl", "al-layl", "lail"], "urdu_script": "لیل"},
    {"number": 93,  "english": "duha",        "arabic": "الضحى",    "transliteration": ["duha", "ad-duha", "doha", "duhaa"], "urdu_script": "ضحیٰ"},
    {"number": 94,  "english": "sharh",       "arabic": "الشرح",    "transliteration": ["sharh", "ash-sharh", "inshirah"], "urdu_script": "شرح"},
    {"number": 95,  "english": "tin",         "arabic": "التين",    "transliteration": ["tin", "at-tin", "teen"], "urdu_script": "تین"},
    {"number": 96,  "english": "alaq",        "arabic": "العلق",    "transliteration": ["alaq", "al-alaq", "alaq"], "urdu_script": "علق"},
    {"number": 97,  "english": "qadr",        "arabic": "القدر",    "transliteration": ["qadr", "al-qadr"], "urdu_script": "قدر"},
    {"number": 98,  "english": "bayyinah",    "arabic": "البينة",   "transliteration": ["bayyinah", "al-bayyinah"], "urdu_script": "بینہ"},
    {"number": 99,  "english": "zalzalah",    "arabic": "الزلزلة",  "transliteration": ["zalzalah", "az-zalzalah", "zilzal"], "urdu_script": "زلزلہ"},
    {"number": 100, "english": "adiyat",      "arabic": "العاديات", "transliteration": ["adiyat", "al-adiyat", "aadiyaat"], "urdu_script": "عادیات"},
    {"number": 101, "english": "qariah",      "arabic": "القارعة",  "transliteration": ["qariah", "al-qariah"], "urdu_script": "قارعہ"},
    {"number": 102, "english": "takathur",    "arabic": "التكاثر",  "transliteration": ["takathur", "at-takathur"], "urdu_script": "تکاثر"},
    {"number": 103, "english": "asr",         "arabic": "العصر",    "transliteration": ["asr", "al-asr", "asr"], "urdu_script": "عصر"},
    {"number": 104, "english": "humazah",     "arabic": "الهمزة",   "transliteration": ["humazah", "al-humazah"], "urdu_script": "ہمزہ"},
    {"number": 105, "english": "fil",         "arabic": "الفيل",    "transliteration": ["fil", "al-fil"], "urdu_script": "فیل"},
    {"number": 106, "english": "quraysh",     "arabic": "قريش",    "transliteration": ["quraysh", "quraish", "al-quraysh"], "urdu_script": "قریش"},
    {"number": 107, "english": "maun",        "arabic": "الماعون",  "transliteration": ["maun", "al-maun", "maa'un"], "urdu_script": "ماعون"},
    {"number": 108, "english": "kawthar",     "arabic": "الكوثر",   "transliteration": ["kawthar", "al-kawthar", "kausar"], "urdu_script": "کوثر"},
    {"number": 109, "english": "kafirun",     "arabic": "الكافرون", "transliteration": ["kafirun", "kafiroon", "al-kafirun"], "urdu_script": "کافرون"},
    {"number": 110, "english": "nasr",        "arabic": "النصر",    "transliteration": ["nasr", "an-nasr"], "urdu_script": "نصر"},
    {"number": 111, "english": "masad",       "arabic": "المسد",    "transliteration": ["masad", "al-masad", "lahab", "tabat"], "urdu_script": "مسد"},
    {"number": 112, "english": "ikhlas",      "arabic": "الإخلاص",  "transliteration": ["ikhlas", "ikhlass", "al-ikhlas"], "urdu_script": "اخلاص"},
    {"number": 113, "english": "falaq",       "arabic": "الفلق",    "transliteration": ["falaq", "al-falaq"], "urdu_script": "فلق"},
    {"number": 114, "english": "nas",         "arabic": "الناس",    "transliteration": ["nas", "an-nas"], "urdu_script": "ناس"},
]


# Build alias → (number, english) lookup. Sort by alias length DESC so longer
# aliases ("an-naml") match before shorter prefixes ("naml").
_ALIAS_TO_SURAH: list[tuple[str, int, str]] = []
for _s in SURAH_DB:
    _aliases: set[str] = set()
    _aliases.add(_s["english"])
    _aliases.add(_s["arabic"])
    _aliases.add(_s["urdu_script"])
    for t in _s["transliteration"]:
        _aliases.add(t)
    for a in _aliases:
        _ALIAS_TO_SURAH.append((a.lower(), _s["number"], _s["english"]))
# Sort longest first so "an-naml" beats "naml"
_ALIAS_TO_SURAH.sort(key=lambda x: -len(x[0]))


def detect_surah_in_query(text: str) -> dict | None:
    """Detect a Surah name in *text* (English / Roman Urdu / Arabic / Urdu-script).

    Returns {"number": int, "english": str, "match": str} or None.

    Strategy:
      1. Try exact substring match against any alias (longest first).
      2. If no exact match, fuzzy-match only the token that immediately
         follows "surah" / "sura" — this is the safest signal that the
         user is naming a surah. cutoff=0.78 catches misspellings like
         "rehmn" → Rahman without grabbing unrelated topic words.
    """
    if not text:
        return None
    norm = text.lower()
    norm = norm.replace("'", "").replace("'", "").replace("`", "")
    # ── 1. Substring match (longest alias first) ──
    for alias, number, english in _ALIAS_TO_SURAH:
        if alias and alias in norm:
            return {"number": number, "english": english, "match": alias}
    # ── 2. Fuzzy match only on the token(s) immediately after "surah"/"sura" ──
    from difflib import get_close_matches
    # Find positions of "surah" / "sura" in the text
    surah_positions = []
    for m in re.finditer(r"\b(surah|sura|surah\s+no\.?)\b", norm):
        surah_positions.append(m.end())
    candidates = [a for (a, n, e) in _ALIAS_TO_SURAH if len(a) >= 3 and a.isascii()]
    for pos in surah_positions:
        # Get the next 1-2 tokens after "surah"
        tail = norm[pos:].strip()
        # Tokenize, ignoring particles like "ka", "ki", "ke", "no", "number"
        skip = {"ka", "ki", "ke", "ko", "se", "me", "mein", "no", "number", "al"}
        toks = [t for t in re.findall(r"[a-z]+", tail) if t not in skip][:3]
        for tok in toks:
            if len(tok) < 3:
                continue
            match = get_close_matches(tok, candidates, n=1, cutoff=0.78)
            if match:
                for alias, number, english in _ALIAS_TO_SURAH:
                    if alias == match[0]:
                        logger.debug("Fuzzy surah match: %r → %r (surah %d)",
                                     tok, match[0], number)
                        return {"number": number, "english": english, "match": match[0]}
    return None


# ── Topic / intent detection ──────────────────────────────────────────────────

# Phrases that mean "give me a summary / overview"
_SUMMARY_TRIGGERS_EN = [
    "summary", "summarize", "summarise", "overview", "introduction", "intro",
    "tell me about", "what is", "what's", "describe", "explain", "brief",
    "khulasa", "خلاصہ",
]
_SUMMARY_TRIGGERS_UR = [
    "ka khulasa", "kya hai", "ke baare mein", "ke baray mein", "ke baray",
    "ka matlab", "ki tafseer", "ki tafseel", "parhna", "padhein", "padho",
    "summarize", "summary", "خلاصہ", "کیا ہے", "کے بارے میں",
]
SUMMARY_INTENT_RE = re.compile(
    r"(khulasa|خلاصہ|tafseer|tafseel|summary|summari[sz]e|overview|"
    r"intro(?:duction)?|what is|what's|describe|explain|brief)",
    re.IGNORECASE,
)


def wants_summary(text: str) -> bool:
    """True if the user is asking for a summary / overview of a surah or topic."""
    t = text.lower()
    if any(trig in t for trig in _SUMMARY_TRIGGERS_EN):
        return True
    if any(trig in t for trig in _SUMMARY_TRIGGERS_UR):
        return True
    return bool(SUMMARY_INTENT_RE.search(t))


# Topic → query rewrite mappings for common Quranic topics.
# When a user query contains one of these topics, we expand the search query
# with the related Arabic/English vocabulary so FAISS retrieves the right
# verses even if the user phrased it colloquially.
TOPIC_EXPANSIONS: dict[str, list[str]] = {
    "wudu":      ["ablution", "wudu", "wudhu", "wuzu", "washing", "water", "face hands arms wipe head feet",
                  "طہارت", "وضو", "Maidah 5:6", "verse of purification", "tahara"],
    "wudhu":     ["ablution", "wudu", "wudhu", "washing", "water", "طہارت", "وضو", "Maidah 5:6"],
    "prayer":    ["prayer", "salah", "salat", "namaz", "establish prayer", "rukoo", "sujud", "qiyam",
                  "نماز", "صلوۃ", "صلاة"],
    "namaz":     ["prayer", "salah", "salat", "namaz", "establish prayer", "rukoo", "sujud", "qiyam",
                  "نماز", "صلوۃ", "صلاة"],
    "salah":     ["prayer", "salah", "salat", "namaz", "نماز", "صلوۃ"],
    "roza":      ["fasting", "sawm", "roza", "roze", "ramadan", "abstain food drink dawn sunset",
                  "روزہ", "صوم", "رمضان"],
    "roze":      ["fasting", "sawm", "roza", "roze", "ramadan", "روزہ", "صوم", "رمضان"],
    "fasting":   ["fasting", "sawm", "roza", "roze", "ramadan", "صوم", "رمضان"],
    "sawm":      ["fasting", "sawm", "roza", "ramadan", "صوم"],
    "zakat":     ["zakat", "zakaat", "charity", "poor", "alms", "wealth purification", "2.5%",
                  "زکوٰۃ", "صدقہ"],
    "zakaat":    ["zakat", "zakaat", "charity", "poor", "alms", "زکوٰۃ", "صدقہ"],
    "charity":   ["zakat", "zakaat", "charity", "poor", "alms", "sadaqah", "زکوٰۃ", "صدقہ"],
    "sadaqah":   ["charity", "sadaqah", "sadaqaat", "poor", "alms", "صدقہ"],
    "hajj":      ["hajj", "pilgrimage", "kaaba", "arafat", "mina", "tawaf", "safaa marwah",
                  "حج", "کعبہ"],
    "sabr":      ["patience", "sabr", "perseverance", "steadfast", "calm", "صبر"],
    "patience":  ["patience", "sabr", "perseverance", "steadfast", "calm", "صبر"],
    "parents":   ["parents", "mother", "father", "kindness to parents", "birr al-walidain",
                  "ماں باپ", "والدین", "walidayn"],
    "mother":    ["parents", "mother", "father", "kindness to parents", "birr al-walidain", "ماں", "والدین"],
    "father":    ["parents", "mother", "father", "kindness to parents", "birr al-walidain", "باپ", "والدین"],
    "tawbah":    ["repentance", "forgiveness", "turn to allah", "mercy", "توبہ", "taubah", "tawba", "touba"],
    "repentance":["repentance", "forgiveness", "turn to allah", "mercy", "توبہ", "taubah", "tawba"],
    "shirk":     ["polytheism", "shirk", "associating partners with allah", "شرک"],
    "jihad":     ["jihad", "struggle", "fighting in allah's cause", "جهاد"],
    "rizq":      ["sustenance", "provision", "rizq", "livelihood", "رزق"],
    "taqwa":     ["piety", "taqwa", "god-consciousness", "righteous", "تقویٰ"],
    "iman":      ["faith", "iman", "belief", "ایمان"],
    "maaf":      ["forgiveness", "magfir", "forgive", "mercy", "rahma", "معافی", "مغفرت"],
    "jannat":    ["paradise", "jannah", "jannat", "heaven", "بهشت", "جنت"],
    "jahannam":  ["hell", "jahannam", "jahannum", "hellfire", "جہنم", "جهنم"],
    "maut":      ["death", "maut", "mawt", "marna", "موت", "موت", "فنا"],
    "qayamat":   ["day of judgment", "qayamat", "qiyamah", "judgment day", "resurrection",
                  "قیامت", "یوم القيامة"],
    "shaitan":   ["satan", "shaitan", "shaytan", "devil", "شيطان", "شیطان"],
    "quran":     ["Quran", "qur'an", "revelation", "book", "قرآن", "القرآن"],
    "rasool":    ["messenger", "rasool", "rasul", "prophet", "nabi", "rasool allah", "نبی", "رسول"],
    "nabi":      ["prophet", "nabi", "messenger", "نبی"],
    "malaika":   ["angels", "malaika", "mala'ika", "ملائکہ", "ملائكة"],
    "jibril":    ["angel Jibreel", "Jibreel", "Gabriel", "jibril", "جبرائیل", "جبريل"],
    "miqat":     ["Miqat", "miqat", "میقات"],
    "qibla":     ["qibla", "direction of prayer", "kaaba", "قبلہ", "كعبة"],
    "halal":     ["halal", "lawful", "permissible", "حلال"],
    "haram":     ["haram", "forbidden", "prohibited", "حرام"],
    "riba":      ["riba", "usury", "interest", "ربا", "ربوی"],
}


# ── Startup self-check: prove the new code paths loaded ──────────────────────
# (Block reserved; the actual print() statements run later, after every function
# is defined. Search for "APP_VERSION banner" below.)


def expand_query_for_topic(text: str) -> str:
    """If *text* mentions a known topic, append related English terms for FAISS.

    This dramatically improves retrieval for colloquial Roman Urdu queries
    like "Wuddu ka zikr Quran mein kahan hai".
    """
    norm = text.lower()
    extras: list[str] = []
    seen_topics: set[str] = set()
    for topic, words in TOPIC_EXPANSIONS.items():
        if topic in seen_topics:
            continue
        # Use word-boundary matching so "roza" doesn't match inside another word
        if re.search(rf"(?<![a-z]){re.escape(topic)}(?![a-z])", norm):
            extras.extend(words)
            seen_topics.add(topic)
    if extras:
        return text + " " + " ".join(extras)
    return text


# ── Greeting Guard ────────────────────────────────────────────────────────────
_GREETINGS = {
    "salam", "salaam", "hi", "hello", "hey", "assalam", "assalamualaikum",
    "السلام", "اسلام", "good morning", "good evening", "good afternoon",
    "thanks", "thank you", "shukriya", "jazakallah", "shukria", "bye",
    "khuda hafiz", "allah hafiz",
}


def _is_greeting(text: str) -> bool:
    t = text.lower().strip()
    return len(t.split()) <= 5 and any(g in t for g in _GREETINGS)


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
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Loading Quran index …")
def load_faiss_index() -> tuple[faiss.Index, list[dict]]:
    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_PATH} / {META_PATH}. "
            "Run `python prepare_data.py` to build it."
        )
    index = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "rb") as f:
        metadata: list[dict] = pickle.load(f)
    logger.info("FAISS index loaded: %d vectors, %d metadata records", index.ntotal, len(metadata))
    if index.ntotal != len(metadata):
        logger.warning(
            "FAISS index / metadata size mismatch: %d vs %d",
            index.ntotal, len(metadata),
        )
    return index, metadata


@st.cache_resource(show_spinner="Connecting to Groq …")
def load_groq_client() -> Groq:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    logger.info("Initialising Groq client (key length=%d)", len(api_key))
    return Groq(api_key=api_key, max_retries=2)


# ── Startup validation ────────────────────────────────────────────────────────

# Hardcoded last-resort list — current Groq free-tier production models.
FREE_TIER_FALLBACKS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "groq/compound-mini",
    "groq/compound",
]


@st.cache_resource(show_spinner="Validating Groq API connectivity …")
def validate_groq_api(_client: Groq) -> tuple[bool, str]:
    """Probe Groq to find a working model. Returns (ok, live_model_name)."""
    api_key_status = "detected" if _get_api_key() else "missing"
    logger.info("Startup check: GROQ_API_KEY=%s | primary=%s | fallback=%s",
                api_key_status, GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK)

    seen: set[str] = set()
    candidates: list[str] = []
    for m in [GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK, *FREE_TIER_FALLBACKS]:
        if m and m not in seen:
            seen.add(m)
            candidates.append(m)

    for candidate in candidates:
        try:
            logger.info("Probing Groq with model=%s", candidate)
            resp = _client.chat.completions.create(
                model=candidate,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
            live = resp.model or candidate
            logger.info("Groq probe OK with model=%s (returned model=%s)", candidate, live)
            st.session_state["_live_groq_model"] = live
            return True, live
        except NotFoundError as exc:
            logger.warning("Model %s not found: %s", candidate, exc)
            continue
        except AuthenticationError as exc:
            logger.error("Groq auth failed: %s", exc)
            return False, f"Authentication failed: {exc}"
        except APIError as exc:
            logger.warning("Groq API error for %s: %s", candidate, exc)
            continue

    tried = ", ".join(f"'{m}'" for m in candidates)
    return False, (
        f"None of the configured models are accessible. Tried: {tried}. "
        "Check that the model names are current and your API key has access. "
        "See https://console.groq.com/docs/models for the current model lineup."
    )


# ── Language helpers ──────────────────────────────────────────────────────────

def is_urdu_or_roman(text: str) -> bool:
    """Detect Urdu script or Roman Urdu so we can translate before embedding."""
    urdu_chars = set("ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنوہھیے")
    if any(c in urdu_chars for c in text):
        return True
    roman_urdu = {
        "mn", "hy", "ha", "hai", "kia", "kya", "bary", "baray", "mein",
        "aaya", "aya", "namaz", "kaha", "aur", "nahi", "hain", "hoon",
        "kr", "tha", "thi", "se", "ko", "ka", "ki", "ke", "ny", "ne",
        "ap", "aap", "mujhe", "hame", "unhe", "ye", "wo", "yeh", "woh",
        "quran", "allah", "rasool", "surah", "ayat", "deen", "islam",
    }
    words = set(text.lower().split())
    return len(words & roman_urdu) >= 2


def _get_live_model() -> str:
    """Return the model that validate_groq_api() confirmed works."""
    return st.session_state.get("_live_groq_model", GROQ_MODEL_PRIMARY)


def _chat_with_fallback(
    client: Groq,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    """Call Groq chat.completions with the same hardened candidate list used at startup."""
    live = _get_live_model()
    seen: set[str] = set()
    candidates: list[str] = []
    for m in [live, GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK, *FREE_TIER_FALLBACKS]:
        if m and m not in seen:
            seen.add(m)
            candidates.append(m)

    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            logger.debug("chat.completions → model=%s", candidate)
            resp = client.chat.completions.create(
                model=candidate,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            used = resp.model or candidate
            if used != _get_live_model():
                logger.info("Live model updated: %s → %s", _get_live_model(), used)
                st.session_state["_live_groq_model"] = used
            return (resp.choices[0].message.content or "").strip(), used
        except NotFoundError as exc:
            last_exc = exc
            logger.warning("Model %s unavailable (%s) — trying next", candidate, exc)
            continue
        except RateLimitError as exc:
            last_exc = exc
            logger.warning("Rate limit on %s: %s — trying next", candidate, exc)
            continue
    assert last_exc is not None
    raise last_exc


def translate_to_english(query: str, client: Groq) -> str:
    """Translate Roman Urdu / Urdu query to English for better FAISS embedding."""
    try:
        text, used = _chat_with_fallback(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translator. If input is Roman Urdu or Urdu script, "
                        "translate to English. If already English, return as-is. "
                        "Return ONLY the translated query, nothing else. Max 15 words."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=60,
            temperature=0.1,
        )
        logger.debug("Translation via %s: %r → %r", used, query, text)
        return text or query
    except Exception as exc:
        logger.warning("Translation failed, using raw query: %s", exc)
        return query


def rewrite_query_for_retrieval(query: str, client: Groq) -> str:
    """Ask the LLM to rewrite a conversational query into a clean search query.

    "Wuddu ka zikr Quran mein kahan hai" → "ablution wudu Quran verses"
    "Surah Naml ka khulasa" → "Surah An-Naml summary"
    """
    try:
        text, _ = _chat_with_fallback(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite user questions into clean English search queries for "
                        "a Quranic verse search engine. Keep Quran-specific terms (Surah names, "
                        "key Arabic/Urdu terms like wudu, namaz, hajj, sabr, taqwa) as-is. "
                        "Translate Roman Urdu or Urdu-script words to English. "
                        "If the question is already in English, return it cleaned up. "
                        "Output ONLY the rewritten search query, no preamble, no quotes. "
                        "Max 25 words."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=60,
            temperature=0.1,
        )
        cleaned = (text or "").strip().strip('"').strip("'")
        if cleaned:
            logger.info("Query rewritten: %r → %r", query, cleaned)
            return cleaned
        return query
    except Exception as exc:
        logger.warning("Query rewrite failed, using raw: %s", exc)
        return query


# ── RAG helpers ───────────────────────────────────────────────────────────────

def search_quran(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
    max_distance: float = 1.5,
    log_scores: bool = True,
) -> list[dict]:
    """Return top_k most relevant verse records for *query*. Logs all scores."""
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)

    if log_scores:
        logger.info("FAISS search: query=%r top_k=%d", query, top_k)
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            if idx == -1:
                continue
            rec = metadata[idx]
            logger.info(
                "  rank %d  dist=%.4f  surah=%s(%d) ayah=%d",
                rank, dist, rec.get("surah_name_english", "?"),
                rec.get("surah_number", 0), rec.get("ayah_number", 0),
            )

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if dist > max_distance:
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


def retrieve_surah_verses(
    surah_number: int,
    metadata: list[dict],
    *,
    max_verses: int | None = None,
) -> list[dict]:
    """Return every verse of a given surah from the metadata, in order.

    If max_verses is set, truncate (useful for short surahs context only).
    """
    matches: list[dict] = []
    for rec in metadata:
        if rec.get("surah_number") != surah_number:
            continue
        matches.append({
            "surah":         rec["surah_name_english"],
            "surah_arabic":  rec["surah_name_arabic"],
            "surah_number":  rec["surah_number"],
            "ayah":          rec["ayah_number"],
            "arabic":        rec["arabic"],
            "urdu":          rec["urdu"],
            "english":       rec["english"],
            "score":         0.0,  # exact-match, not a distance
        })
    # sort by ayah number
    matches.sort(key=lambda r: r["ayah"])
    if max_verses is not None:
        matches = matches[:max_verses]
    logger.info("Retrieved %d verses from Surah %d (%s) by exact match",
                len(matches), surah_number, matches[0]["surah"] if matches else "?")
    return matches


def build_context(results: list[dict], *, max_chars: int = 12000) -> str:
    """Format search results into an LLM-readable context block.

    *max_chars* protects against accidentally stuffing 6000 verses into the prompt.
    """
    lines: list[str] = []
    total = 0
    for i, r in enumerate(results, 1):
        block = (
            f"[Verse {i}] Surah {r['surah']} ({r['surah_number']}), Ayah {r['ayah']}\n"
            f"Arabic: {r['arabic']}\n"
            f"English: {r['english']}\n"
            f"Urdu: {r['urdu']}\n"
            f"{'-' * 60}\n"
        )
        if total + len(block) > max_chars and lines:
            logger.info("Context truncated at %d verses to stay under %d chars",
                        len(lines) - 1, max_chars)
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines)


def build_system_prompt(*, summary_mode: bool, surah: dict | None) -> str:
    """Build a context-aware system prompt.

    *summary_mode*  — True when the user asked for a summary/overview
    *surah*         — {"number", "english"} when a specific surah was detected
    """
    base = """You are a knowledgeable Quran AI Assistant created by Ubaid ur Rehman,
an Aalim (Islamic Scholar) and AI Developer. The Urdu translation used is by
Maulana Fateh Muhammad Jalandhri رحمه الله (Deobandi school), and the English
translation is Saheeh International.

CORE BEHAVIOR:
1. Use the provided Quranic verses as your PRIMARY source. They are real verses
   from the indexed Quran, not fabricated.
2. Always cite the Surah name and Ayah number when you reference a verse.
3. Include the Arabic text, English translation, and Urdu translation in your answer.
4. Be scholarly, respectful, and humble in tone.
5. If the user writes in Urdu or Roman Urdu, respond in Urdu. Otherwise respond
   in English.
6. If the verses you were given are insufficient, you may briefly add
   well-known classical context (e.g. "Surah X is generally regarded as a
   Meccan/Medinan surah, focusing on Y themes") but stay anchored to the verses
   you actually have. Do NOT invent verse text or numbers.
7. Format Arabic verses clearly, with a separate line for each translation.
8. If the user sends only a greeting, respond warmly and ask them to pose a
   Quran-related question."""

    if surah is not None and summary_mode:
        return base + f"""

CURRENT TASK — SUMMARIZE SURAH {surah['number']} ({surah['english'].title()}):
You have been given the complete text of this surah (or as much as fits in the
context window). Your job is to produce a CONCISE, FAITHFUL summary that
covers:
  • The surah's main theme(s) and message(s)
  • The historical context (Meccan / Medinan, approximate period) if you can tell
    from the verses
  • 2-4 KEY verses that best represent the surah's message — quote them in full
    (Arabic + English + Urdu) with their Surah:Ayah citation
  • Practical lessons a reader can take away

Write in the user's language (Urdu if they asked in Urdu, English otherwise)."""
    if surah is not None:
        return base + f"""

CURRENT TASK — ANSWER ABOUT SURAH {surah['number']} ({surah['english'].title()}):
The user asked something specific about this surah. Answer their question
using the verses you were given. Quote the most relevant verses in full."""
    if summary_mode:
        return base + """

CURRENT TASK — SUMMARIZE A TOPIC:
The user asked for a summary of a Quranic topic. Synthesize the provided
verses into a clear, organized summary. Quote 2-4 representative verses in
full (Arabic + English + Urdu). If the verses cover multiple sub-topics,
organize your answer with subheadings."""
    return base


def get_ai_response(
    query: str,
    context: str,
    client: Groq,
    *,
    summary_mode: bool = False,
    surah: dict | None = None,
) -> str:
    """Send query + context to Groq and return the assistant's reply."""
    system_prompt = build_system_prompt(summary_mode=summary_mode, surah=surah)
    try:
        text, used = _chat_with_fallback(
            client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}\n\nRelevant Quran verses:\n{context}"},
            ],
            max_tokens=2048,
            temperature=0.3,
        )
        logger.info("Answer generated via model=%s (len=%d, summary_mode=%s, surah=%s)",
                    used, len(text), summary_mode, surah)
        return text
    except RateLimitError as exc:
        logger.warning("Rate limit on both models: %s", exc)
        return "⏳ AI is resting for a moment. Please wait 1-2 minutes and try again."
    except NotFoundError as exc:
        logger.error("No model available: %s", exc)
        return (
            "⚠️ The AI model is currently unavailable on Groq. "
            f"Tried '{GROQ_MODEL_PRIMARY}' and '{GROQ_MODEL_FALLBACK}'. "
            "Please check https://console.groq.com/docs/models for current model IDs."
        )
    except APIConnectionError as exc:
        logger.error("Groq connection error: %s", exc)
        return "⚠️ Cannot reach Groq API. Check your network connection and try again."
    except APIStatusError as exc:
        logger.exception("Groq API status error")
        return f"⚠️ AI service error (HTTP {exc.status_code}). Please try again."
    except APIError as exc:
        logger.exception("Groq API error")
        return f"⚠️ An error occurred: {exc}\n\nPlease try again."
    except Exception as exc:
        logger.exception("Unexpected error in get_ai_response")
        return f"⚠️ An error occurred: {exc}\n\nPlease try again."


# ── Sidebar ───────────────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "What does the Quran say about patience?",
    "Sabr ke baare mein kya aaya hai?",
    "Tell me about Surah Al-Fatiha",
    "What are the verses about charity?",
    "Namaz ke baare mein ayaat",
    "What does the Quran say about parents?",
    "Surah Naml ka khulasa likho",
    "Wuddu ka zikr Quran mein kahan hai?",
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

    if FAISS_PATH.exists():
        st.markdown("### 📊 Stats")
        st.markdown(
            f"- **Verses indexed:** 6,236\n"
            f"- **Live model:** `{_get_live_model()}`\n"
            f"- **Configured primary:** `{GROQ_MODEL_PRIMARY}`\n"
            f"- **Configured fallback:** `{GROQ_MODEL_FALLBACK}`"
        )
        st.divider()

    st.markdown("### 🔗 Links")
    st.markdown(
        "- [GitHub](https://github.com/Ub207)\n"
        "- [Portfolio](https://github.com/Ub207)"
    )

    st.divider()
    st.caption(f"**App version:** `{APP_VERSION}`")


# ── Guard: API key ────────────────────────────────────────────────────────────

if not _get_api_key():
    st.error(
        "**GROQ_API_KEY not found.**\n\n"
        "For local development, create a `.env` file with:\n"
        "```\nGROQ_API_KEY=your_key_here\n```\n\n"
        "For Hugging Face Spaces, add it in **Settings → Repository secrets**.\n"
        "For Streamlit Cloud, add it in **Settings → Secrets** as:\n"
        "```toml\nGROQ_API_KEY = \"gsk_...\"\n```\n\n"
        "Get a free key at [console.groq.com](https://console.groq.com)."
    )
    logger.error("GROQ_API_KEY missing — aborting startup.")
    st.stop()


# ── Guard: FAISS index ────────────────────────────────────────────────────────

if not FAISS_PATH.exists() or not META_PATH.exists():
    st.error(
        "**FAISS index not found.**\n\n"
        "Please run the data preparation script first:\n"
        "```bash\npython prepare_data.py\n```\n\n"
        "This downloads the Quran data and builds the search index."
    )
    logger.error("FAISS index missing — aborting startup.")
    st.stop()


# ── APP_VERSION banner (runs once per container start, AFTER all defs) ────────
# If you see this in the Space's runtime logs, the new code is loaded.
print("=" * 70)
print(f"APP_VERSION: {APP_VERSION}")
print(f"Surah aliases loaded: {len(_ALIAS_TO_SURAH)}")
print(f"Surah DB entries: {len(SURAH_DB)}")
print("Summary mode enabled")
print(f"Topic expansion enabled ({len(TOPIC_EXPANSIONS)} topics)")
print(f"detect_surah_in_query: {detect_surah_in_query.__name__}")
print(f"retrieve_surah_verses: {retrieve_surah_verses.__name__}")
print(f"expand_query_for_topic: {expand_query_for_topic.__name__}")
print("=" * 70)
logger.info("APP_VERSION: %s  |  aliases=%d  |  topics=%d",
            APP_VERSION, len(_ALIAS_TO_SURAH), len(TOPIC_EXPANSIONS))


# ── Load resources ────────────────────────────────────────────────────────────

try:
    embed_model       = load_embedding_model()
    faiss_index, meta = load_faiss_index()
    groq_client       = load_groq_client()
except Exception as exc:
    logger.exception("Failed to load resources")
    st.error(f"**Startup error:** {exc}")
    st.stop()


# ── Startup: validate API connectivity ────────────────────────────────────────

ok, info = validate_groq_api(groq_client)
if not ok:
    st.error(
        f"**Cannot reach Groq API.**\n\n{info}\n\n"
        "Verify your `GROQ_API_KEY` and that the configured model names "
        "are current at [console.groq.com/docs/models](https://console.groq.com/docs/models)."
    )
    logger.error("Groq validation failed: %s", info)
    st.stop()

live_model = info
st.session_state["_live_groq_model"] = live_model
logger.info("Startup validation passed; live model=%s", live_model)
logger.info("FAISS metadata loaded: %d verses across 114 surahs", len(meta))

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
st.caption(f"Connected to Groq · live model: `{live_model}` · {len(meta):,} verses indexed")
st.caption(f"App version: `{APP_VERSION}` · {len(SURAH_DB)} surahs · {len(TOPIC_EXPANSIONS)} topic expansions loaded")
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
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    if _is_greeting(user_input):
        greeting_reply = (
            "وعليكم السلام ورحمة الله وبركاته! 😊\n\n"
            "Quran ke baare mein koi sawaal poochein — "
            "main Urdu aur English dono mein jawab dene ki koshish karunga.\n\n"
            "**Misal ke taur par:**\n"
            "- *Sabr ke baare mein Quran mein kya aaya hai?*\n"
            "- *What does the Quran say about parents?*\n"
            "- *Surah Al-Fatiha ki tafseer bataein*\n"
            "- *Surah Naml ka khulasa likho*\n"
            "- *Wuddu ka zikr Quran mein kahan hai?*"
        )
        with st.chat_message("assistant"):
            st.markdown(greeting_reply)
        st.session_state.messages.append({
            "role":    "assistant",
            "content": greeting_reply,
        })
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching Quran & generating answer…"):
            results: list[dict] = []
            detected_surah: dict | None = None
            is_summary = False
            try:
                # ── 1. Detect a specific Surah in the query ───────────────────
                detected_surah = detect_surah_in_query(user_input)
                is_summary = wants_summary(user_input)

                logger.info(
                    "Query received: %r  detected_surah=%s  wants_summary=%s",
                    user_input, detected_surah, is_summary,
                )

                if detected_surah is not None:
                    # ── 2a. Surah-specific path: return ALL verses of that surah
                    results = retrieve_surah_verses(
                        detected_surah["number"], meta,
                        max_verses=300 if is_summary else 50,
                    )
                else:
                    # ── 2b. Generic query path: rewrite + topic-expand + FAISS
                    search_query = user_input

                    # If Roman Urdu / Urdu script, first translate to English
                    if is_urdu_or_roman(user_input):
                        search_query = translate_to_english(user_input, groq_client)
                        logger.info("Roman/Urdu query translated: %r → %r",
                                    user_input, search_query)

                    # Expand known topic vocabulary so FAISS retrieves the right
                    # verses for colloquial phrasings like "Wuddu ka zikr…"
                    search_query = expand_query_for_topic(search_query)

                    # LLM-based query rewrite: cleans the query for the embedder
                    search_query = rewrite_query_for_retrieval(search_query, groq_client)

                    results = search_quran(
                        search_query, faiss_index, meta, embed_model,
                        top_k=8, max_distance=1.6, log_scores=True,
                    )

                if not results:
                    answer = (
                        "🔍 **Mujhe is sawaal se mutaliq koi Quranic ayat nahi mili.**\n\n"
                        "Yeh Quran se bahar ka sawaal lag raha hai, ya query itni general hai "
                        "ke relevant verses nahi mile.\n\n"
                        "Meherbani karke Quran ke baare mein specific sawaal poochein, jaise:\n"
                        "- *What does Quran say about patience?*\n"
                        "- *Surah Al-Fatiha ki tafseer bataein*\n"
                        "- *Waalidain ke baare mein ayaat*"
                    )
                else:
                    context = build_context(results)
                    answer = get_ai_response(
                        user_input, context, groq_client,
                        summary_mode=is_summary,
                        surah=detected_surah,
                    )

            except Exception as exc:
                logger.exception("Chat pipeline error")
                answer = f"⚠️ An error occurred: {exc}\n\nPlease try again."

        st.markdown(answer, unsafe_allow_html=True)

        if results:
            with st.expander("📜 View Source Verses", expanded=False):
                # Show at most 20 verses in the UI to keep it readable
                for v in results[:20]:
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
                if len(results) > 20:
                    st.caption(f"… and {len(results) - 20} more verses (see logs for full list)")

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
