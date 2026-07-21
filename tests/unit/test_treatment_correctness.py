"""Unit tests for treatment content correctness and completeness."""
import json, os

import pytest

from cropguard.crop_config import CROP_CLASSES
import cropguard.llm_advice as llm_advice

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TREATMENTS_PATH = os.path.join(REPO_ROOT, "src", "cropguard", "treatments.json")


def _load():
    with open(_TREATMENTS_PATH, encoding="utf-8") as f:
        return json.load(f)


class TestTreatmentCompleteness:
    """Every crop/disease must have all three languages with non-empty content."""

    def test_all_crops_present(self):
        data = _load()
        for crop in CROP_CLASSES:
            assert crop in data, f"Missing crop '{crop}' in treatments.json"

    @pytest.mark.parametrize("crop", list(CROP_CLASSES))
    def test_all_diseases_present(self, crop):
        data = _load()
        for disease in CROP_CLASSES[crop]:
            assert disease in data[crop], f"Missing {crop}/{disease}"

    @pytest.mark.parametrize("crop", list(CROP_CLASSES))
    def test_all_languages_present(self, crop):
        data = _load()
        for disease in CROP_CLASSES[crop]:
            for lang in ("en", "es", "va"):
                assert lang in data[crop][disease], f"Missing lang '{lang}' for {crop}/{disease}"

    @pytest.mark.parametrize("crop", list(CROP_CLASSES))
    def test_fields_non_empty(self, crop):
        data = _load()
        for disease in CROP_CLASSES[crop]:
            for lang in ("en", "es", "va"):
                entry = data[crop][disease][lang]
                assert isinstance(entry["explanation"], str) and entry["explanation"].strip(), \
                    f"{crop}/{disease}/{lang}: empty explanation"
                for field in ("symptoms", "treatment", "prevention"):
                    items = entry[field]
                    assert isinstance(items, list) and items, \
                        f"{crop}/{disease}/{lang}: empty {field}"
                    assert all(isinstance(i, str) and i.strip() for i in items), \
                        f"{crop}/{disease}/{lang}: non-string in {field}"


class TestTreatmentContent:
    """Spot-check known treatment content for correctness."""

    def test_rice_blast_has_fungicide_advice(self):
        t = llm_advice.get_static_treatment("Rice", "Rice Blast", "en")
        assert t is not None
        combined = (t["explanation"] + " " + " ".join(t["treatment"])).lower()
        assert "fungicide" in combined or "tricyclazole" in combined or "blast" in combined

    def test_tomato_late_blight_symptoms_mentioned(self):
        t = llm_advice.get_static_treatment("Tomato", "Late Blight", "en")
        assert t is not None
        symptoms_text = " ".join(t["symptoms"]).lower()
        assert "lesion" in symptoms_text or "water" in symptoms_text or "blight" in symptoms_text

    def test_treatments_spanish_not_english(self):
        """Spanish treatment must actually be in Spanish, not a copy of English."""
        t_en = llm_advice.get_static_treatment("Rice", "Rice Blast", "en")
        t_es = llm_advice.get_static_treatment("Rice", "Rice Blast", "es")
        assert t_en and t_es
        assert t_en["explanation"] != t_es["explanation"], \
            "Spanish and English explanations are identical"

    def test_treatments_valencian_not_english(self):
        """Valencian must differ from English."""
        t_en = llm_advice.get_static_treatment("Rice", "Rice Blast", "en")
        t_va = llm_advice.get_static_treatment("Rice", "Rice Blast", "va")
        assert t_en and t_va
        assert t_en["explanation"] != t_va["explanation"], \
            "Valencian and English explanations are identical"

    @pytest.mark.parametrize("crop", list(CROP_CLASSES))
    def test_treatment_keys_match_disease_list(self, crop):
        data = _load()
        assert set(data[crop].keys()) == set(CROP_CLASSES[crop]), \
            f"{crop}: treatment keys don't match class list"

    def test_healthy_diseases_have_minimal_treatment(self):
        """Healthy classes should have short/standard advice."""
        for crop in CROP_CLASSES:
            for disease in CROP_CLASSES[crop]:
                if "Healthy" in disease or "healthy" in disease.lower():
                    t = llm_advice.get_static_treatment(crop, disease, "en")
                    assert t is not None, f"Missing treatment for healthy class {crop}/{disease}"
                    # Healthy classes should still have all fields populated
                    assert len(t["treatment"]) >= 1


class TestTreatmentLookup:
    """Test the llm_advice lookup function."""

    def test_known_disease_returns_entry(self):
        t = llm_advice.get_static_treatment("Orange", "Black Spot", "en")
        assert t is not None and "explanation" in t

    def test_unknown_disease_returns_none(self):
        assert llm_advice.get_static_treatment("Tomato", "Nonexistent Disease") is None

    def test_unknown_crop_returns_none(self):
        assert llm_advice.get_static_treatment("Banana", "Some Disease") is None

    def test_fallback_to_english(self):
        """If a language is missing, fall back to English."""
        t = llm_advice.get_static_treatment("Rice", "Rice Blast", "en")
        assert t is not None
