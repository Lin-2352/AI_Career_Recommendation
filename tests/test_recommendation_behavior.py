import math
import unittest

from backend.career_recommender.modeling import load_artifact
from backend.career_recommender.recommendation import (
    career_alignment_score,
    extract_profile_signals,
    profile_from_inputs,
    rank_profile_recommendations,
)
from frontend.ui_helpers import confidence_band


class RecommendationBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = load_artifact()

    def ranked(self, age, education, skills, interests, extra_context=""):
        profile = profile_from_inputs(age, education, skills, interests, extra_context)
        return rank_profile_recommendations(self.artifact, profile, skills, interests, extra_context, top_n=5)

    def test_prediction_values_are_well_formed(self):
        results = self.ranked(24, "Bachelor's", "Python, SQL, machine learning", "AI, data science", "Technical")
        self.assertEqual(len(results), 5)
        for item in results:
            self.assertIn("career", item)
            self.assertTrue(0 <= item["confidence"] <= 1)
            self.assertTrue(0 <= item["profile_alignment"] <= 1)
            self.assertTrue(math.isfinite(item["fit_score"]))

    def test_machine_learning_profile_ranks_data_roles_first(self):
        results = self.ranked(
            24,
            "Bachelor's",
            "Python, SQL, machine learning, statistics",
            "AI, data science",
            "Technical Technology",
        )
        top_three = {item["career"] for item in results[:3]}
        self.assertTrue({"Data Scientist", "Machine Learning Engineer"}.intersection(top_three))

    def test_finance_profile_prefers_financial_analyst(self):
        results = self.ranked(
            22,
            "Bachelor's",
            "financial modeling, accounting, excel, risk analysis",
            "finance, banking, markets",
            "Business Finance",
        )
        self.assertEqual(results[0]["career"], "Financial Analyst")

    def test_design_profile_prefers_ux_designer(self):
        results = self.ranked(
            21,
            "Design",
            "graphic design, figma, typography, user research",
            "design, UX, creative work",
            "Design",
        )
        self.assertEqual(results[0]["career"], "UX Designer")

    def test_healthcare_profile_prefers_doctor(self):
        results = self.ranked(
            18,
            "Secondary school",
            "biology, chemistry, strong biology, strong chemistry",
            "healthcare, medicine, patient care",
            "Healthcare",
        )
        self.assertEqual(results[0]["career"], "Doctor")

    def test_unknown_profile_is_not_overstated(self):
        results = self.ranked(30, "Diploma", "asdf qwer zzzz", "lorem ipsum unknown")
        best = results[0]
        displayed_total = sum(item["fit_score"] for item in results) or 1
        fit_share = best["fit_score"] / displayed_total
        self.assertEqual(confidence_band(fit_share, best["confidence"], best["profile_alignment"]), "Exploratory match")

    def test_signal_extraction_is_deduplicated(self):
        signals = extract_profile_signals("Cloud, cloud computing, Python", "Cloud infrastructure and AI")
        self.assertEqual(len(signals), len(set(signals)))

    def test_alignment_handles_known_and_unknown_careers(self):
        self.assertGreater(career_alignment_score("Data Scientist", "Python SQL statistics machine learning"), 0)
        self.assertEqual(career_alignment_score("Unknown Career", "Python SQL"), 0)


if __name__ == "__main__":
    unittest.main()
