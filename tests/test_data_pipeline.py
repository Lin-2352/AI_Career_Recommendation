import unittest

import pandas as pd

from backend.career_recommender.data_pipeline import build_master_dataset


class DataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = build_master_dataset()

    def test_master_dataset_has_expected_shape(self):
        self.assertGreaterEqual(len(self.master), 3000)
        self.assertGreaterEqual(self.master["target_career"].nunique(), 50)

    def test_sources_are_tracked(self):
        sources = set(self.master["source"].dropna())
        self.assertIn("career_guidance", sources)
        self.assertIn("career_recommendation", sources)
        self.assertIn("student_scores", sources)

    def test_required_training_fields_are_populated(self):
        required = ["education_level", "technical_skills", "professional_interests", "target_career", "profile_text"]
        for column in required:
            self.assertFalse(self.master[column].isna().any(), column)
            self.assertFalse((self.master[column].astype(str).str.strip() == "").any(), column)

    def test_sanitized_data_has_no_direct_identifiers(self):
        sanitized = pd.read_csv("data/raw/student_scores_sanitized.csv", nrows=5)
        blocked_columns = {"id", "first_name", "last_name", "email", "gender"}
        self.assertTrue(blocked_columns.isdisjoint(set(sanitized.columns)))
        self.assertFalse(self.master["profile_text"].astype(str).str.contains("@", regex=False).any())


if __name__ == "__main__":
    unittest.main()
