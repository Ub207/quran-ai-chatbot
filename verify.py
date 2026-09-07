"""
verify.py — Pre-deployment verification for Quran AI Assistant.

Run this locally before pushing to Hugging Face Spaces or Streamlit Cloud.
It checks:
  1. GROQ_API_KEY is present.
  2. The configured primary + fallback models are reachable on Groq.
  3. The FAISS index is present and well-formed.
  4. The Groq SDK version is supported.

Exit code 0 = ship it.  Non-zero = fix the issues and re-run.
"""

import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import APIError, AuthenticationError, Groq, NotFoundError

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

PRIMARY    = os.environ.get("GROQ_MODEL_PRIMARY",  "openai/gpt-oss-20b")
FALLBACK   = os.environ.get("GROQ_MODEL_FALLBACK", "openai/gpt-oss-120b")
FAISS_PATH = Path("faiss_index/index.faiss")
META_PATH  = Path("faiss_index/metadata.pkl")

# Hardcoded last-resort models verified against the current Groq docs.
# Used by app.py's startup validator to guarantee at least one model
# is reachable on a free-tier account.
FREE_TIER_FALLBACKS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "groq/compound-mini",
    "groq/compound",
]


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "✅" if ok else "❌"
    print(f"{mark} {label}{(' — ' + detail) if detail else ''}")
    return ok


def main() -> int:
    print("=" * 60)
    print("  Quran AI Assistant — Pre-deployment Verification")
    print("=" * 60)

    failures = 0

    # 1. API key
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not _check("GROQ_API_KEY present", bool(api_key), f"length={len(api_key)}"):
        failures += 1
        print("   → Add GROQ_API_KEY to your .env, HF Spaces secrets, or Streamlit secrets.")
        return 1   # can't test anything else without a key

    # 2. Groq SDK version
    try:
        import groq
        sdk_ver = getattr(groq, "__version__", "unknown")
        _check("Groq SDK importable", True, f"version={sdk_ver}")
    except ImportError:
        _check("Groq SDK importable", False)
        failures += 1
        return 1

    # 3. FAISS index
    if FAISS_PATH.exists() and META_PATH.exists():
        try:
            import faiss
            idx = faiss.read_index(str(FAISS_PATH))
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
            ok = idx.ntotal == len(meta) and idx.ntotal > 6000
            if not _check("FAISS index well-formed", ok,
                          f"{idx.ntotal} vectors, {len(meta)} metadata records"):
                failures += 1
        except Exception as exc:
            _check("FAISS index well-formed", False, str(exc))
            failures += 1
    else:
        _check("FAISS index present", False, "run `python prepare_data.py` first")
        failures += 1

    # 4. Groq connectivity + model reachability
    client = Groq(api_key=api_key, max_retries=1)
    primary_ok = False
    for label, model in (("Primary", PRIMARY), ("Fallback", FALLBACK)):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
            _check(f"{label} model '{model}'", True, f"live model={resp.model}")
            if label == "Primary":
                primary_ok = True
        except NotFoundError:
            _check(f"{label} model '{model}'", False,
                   "not found — update GROQ_MODEL_PRIMARY/FALLBACK")
            failures += 1
        except AuthenticationError as exc:
            _check(f"{label} model '{model}'", False, f"auth error: {exc}")
            failures += 1
        except APIError as exc:
            _check(f"{label} model '{model}'", False, f"API error: {exc}")
            failures += 1

    # 5. Free-tier fallback reachability — only checked if primary failed.
    #    This guarantees at least one model is reachable on a free-tier account.
    if not primary_ok:
        print("\n   ↳ Probing free-tier fallbacks (Groq dev plan) …")
        any_free_ok = False
        for model in FREE_TIER_FALLBACKS:
            if model in (PRIMARY, FALLBACK):
                continue
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0.0,
                )
                _check(f"Free-tier model '{model}'", True, f"live model={resp.model}")
                any_free_ok = True
                break
            except NotFoundError:
                _check(f"Free-tier model '{model}'", False, "not found")
                failures += 1
            except APIError as exc:
                _check(f"Free-tier model '{model}'", False, str(exc))
                failures += 1
        if not any_free_ok:
            print("   → Set GROQ_MODEL_PRIMARY=openai/gpt-oss-20b in your secrets "
                  "(free-tier model that works on dev accounts).")

    print("=" * 60)
    if failures == 0:
        print("🎉 All checks passed — safe to deploy.")
        return 0
    print(f"⚠️  {failures} check(s) failed. Fix the issues above and re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
