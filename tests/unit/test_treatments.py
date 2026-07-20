"""Unit tests for treatments.json — static advice completeness in all languages."""
import json, os

from cropguard.crop_config import CROP_CLASSES
import cropguard.llm_advice as llm_advice

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TREATMENTS_PATH = os.path.join(REPO_ROOT, "src", "cropguard", "treatments.json")


def _load():
    with open(_TREATMENTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_every_disease_present_in_three_languages():
    data = _load()
    for crop, classes in CROP_CLASSES.items():
        assert crop in data, f"missing crop {crop}"
        for disease in classes:
            assert disease in data[crop], f"missing {crop}/{disease}"
            for lang in ("en", "es", "va"):
                assert lang in data[crop][disease], f"missing lang {lang} for {crop}/{disease}"


def test_fields_non_empty():
    data = _load()
    for crop, diseases in data.items():
        for disease, entry in diseases.items():
            for lang, content in entry.items():
                assert isinstance(content["explanation"], str) and content["explanation"].strip()
                for field in ("symptoms", "treatment", "prevention"):
                    items = content[field]
                    assert isinstance(items, list) and items, f"{crop}/{disease}/{lang}/{field} empty"
                    assert all(isinstance(i, str) and i.strip() for i in items)


def test_get_static_treatment_lookup_and_fallback():
    t = llm_advice.get_static_treatment("Rice", "Rice Blast", "va")
    assert t and "explanation" in t
    # Unknown disease → None
    assert llm_advice.get_static_treatment("Rice", "Not A Disease") is None
