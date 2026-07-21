"""
CropGuard — treatment advice helper (shared by server.py and app.py).

Two tiers of advice:
1. Static recommendations from treatments.json (always available, no key needed).
2. Optional AI follow-up answers via:
   - Groq (GROQ_API_KEY env var, OpenAI-compatible API)
   - Google Gemini free tier (google-genai package, GEMINI_API_KEY env var)
   - OpenCode Go (OPENCODE_API_KEY env var, OpenAI-compatible API)
   The AI receives the static content as grounding context and answers in the user's language.
"""
import os, json
from pathlib import Path

# Load .env from project src/ directory (one level up from cropguard package)
_env_path = Path(__file__).resolve().parent.parent / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_env_path)
except ImportError:
    pass

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

LANG_NAMES = {"en": "English", "es": "Spanish", "va": "Valencian"}
VALID_LANGS = tuple(LANG_NAMES)

# Provider constants
PROVIDER_GROQ = "groq"
PROVIDER_GEMINI = "gemini"
PROVIDER_OPENCODE = "opencode"
DEFAULT_PROVIDER = os.environ.get("CROPGUARD_AI_PROVIDER", PROVIDER_GROQ)

GEMINI_MODEL = "gemini-2.0-flash"
OPENCODE_MODEL = "qwen3.5-plus"
OPENCODE_API_URL = "https://opencode.ai/zen/go/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

_treatments = None
_gemini_client = "unset"


def _load_treatments():
    global _treatments
    if _treatments is None:
        with open(os.path.join(WORK_DIR, "treatments.json"), encoding="utf-8") as f:
            _treatments = json.load(f)
    return _treatments


def get_static_treatment(crop, disease, lang="en"):
    """Return {explanation, symptoms, treatment, prevention} or None if unknown.
    Falls back to English if the requested language is missing."""
    entry = _load_treatments().get(crop, {}).get(disease)
    if entry is None:
        return None
    return entry.get(lang) or entry.get("en")


def _get_gemini_client():
    """Lazy Gemini client; None if the package or API key is missing."""
    global _gemini_client
    if _gemini_client != "unset":
        return _gemini_client
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        _gemini_client = None
        return None
    try:
        from google import genai
    except ImportError:
        _gemini_client = None
        return None
    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_opencode_key():
    """Return OpenCode API key or None if not set."""
    return os.environ.get("OPENCODE_API_KEY", "") or None


def _get_groq_key():
    """Return Groq API key or None if not set."""
    return os.environ.get("GROQ_API_KEY", "") or None


def build_prompt(crop, disease, question, lang):
    static = get_static_treatment(crop, disease, lang) or get_static_treatment(crop, disease, "en") or {}
    context = json.dumps(static, ensure_ascii=False)
    language = LANG_NAMES.get(lang, "English")
    return (
        f"You are an expert agronomist specializing in {crop} cultivation. "
        f"A farmer's {crop} plant has been diagnosed with: {disease}. "
        f"Here is the reference information about this disease (JSON): {context}\n\n"
        f"Answer the farmer's follow-up question in {language}. "
        f"Keep it concise (max ~150 words), practical, and prefer ecological/home remedies. "
        f"If the problem sounds severe, suggest consulting a professional agronomist.\n\n"
        f"Question: {question}"
    )


def _ask_gemini(crop, disease, question, lang):
    """Ask Gemini. Returns answer string or raises."""
    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client unavailable (set GEMINI_API_KEY)")
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_prompt(crop, disease, question, lang),
    )
    return resp.text


def _ask_opencode(crop, disease, question, lang):
    """Ask OpenCode Go via OpenAI-compatible API. Returns answer string or raises."""
    import requests as _requests

    api_key = _get_opencode_key()
    if not api_key:
        raise RuntimeError("OpenCode API key unavailable (set OPENCODE_API_KEY)")
    resp = _requests.post(
        OPENCODE_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": OPENCODE_MODEL,
            "messages": [{"role": "user", "content": build_prompt(crop, disease, question, lang)}],
            "max_tokens": 512,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _ask_groq(crop, disease, question, lang):
    """Ask Groq via OpenAI-compatible API. Returns answer string or raises."""
    import requests as _requests

    api_key = _get_groq_key()
    if not api_key:
        raise RuntimeError("Groq API key unavailable (set GROQ_API_KEY)")
    resp = _requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": build_prompt(crop, disease, question, lang)}],
            "max_tokens": 512,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def ask_followup(crop, disease, question, lang="en", provider=None):
    """Ask the AI a follow-up question about a diagnosed disease.

    Returns {"answer": str|None, "fallback": dict|None, "note": str|None}.
    When the AI is unavailable, answer is None and fallback holds the static
    treatment so callers can still show something useful."""
    provider = provider or DEFAULT_PROVIDER
    try:
        if provider == PROVIDER_GROQ:
            answer = _ask_groq(crop, disease, question, lang)
        elif provider == PROVIDER_GEMINI:
            answer = _ask_gemini(crop, disease, question, lang)
        elif provider == PROVIDER_OPENCODE:
            answer = _ask_opencode(crop, disease, question, lang)
        else:
            return {
                "answer": None,
                "fallback": get_static_treatment(crop, disease, lang),
                "note": f"Unknown provider: {provider}",
            }
        return {"answer": answer, "fallback": None, "note": None}
    except Exception as e:
        return {
            "answer": None,
            "fallback": get_static_treatment(crop, disease, lang),
            "note": f"AI error ({provider}): {e}",
        }
