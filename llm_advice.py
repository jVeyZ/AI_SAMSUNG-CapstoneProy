"""
CropGuard — treatment advice helper (shared by server.py and app.py).

Two tiers of advice:
1. Static recommendations from treatments.json (always available, no key needed).
2. Optional AI follow-up answers via the Gemini free tier (google-genai package,
   GEMINI_API_KEY env var). The AI receives the static content as grounding
   context and answers in the user's language.
"""
import os, json

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

LANG_NAMES = {"en": "English", "es": "Spanish", "va": "Valencian"}
VALID_LANGS = tuple(LANG_NAMES)

GEMINI_MODEL = "gemini-2.0-flash"

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


def ask_followup(crop, disease, question, lang="en"):
    """Ask the AI a follow-up question about a diagnosed disease.

    Returns {"answer": str|None, "fallback": dict|None, "note": str|None}.
    When the AI is unavailable, answer is None and fallback holds the static
    treatment so callers can still show something useful."""
    client = _get_gemini_client()
    if client is None:
        return {
            "answer": None,
            "fallback": get_static_treatment(crop, disease, lang),
            "note": "AI unavailable (set GEMINI_API_KEY) — showing stored recommendations.",
        }
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=build_prompt(crop, disease, question, lang),
        )
        return {"answer": resp.text, "fallback": None, "note": None}
    except Exception as e:
        return {
            "answer": None,
            "fallback": get_static_treatment(crop, disease, lang),
            "note": f"AI error: {e}",
        }
