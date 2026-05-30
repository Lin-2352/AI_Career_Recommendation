import unittest

from frontend.ui_helpers import EDUCATION_PLACEHOLDER, confidence_band, split_terms, validate_inputs


class UiHelperTests(unittest.TestCase):
    def test_split_terms_handles_common_separators(self):
        self.assertEqual(split_terms("Python, SQL / AI\nData"), ["Python", "SQL", "AI", "Data"])

    def test_required_validation_catches_empty_profile(self):
        errors = validate_inputs(EDUCATION_PLACEHOLDER, "", "")
        self.assertEqual(len(errors), 3)

    def test_required_validation_catches_numeric_noise(self):
        errors = validate_inputs("Bachelor's", "123, 456", "789")
        self.assertIn("Add at least two meaningful skills.", errors)
        self.assertIn("Add at least one meaningful professional interest.", errors)

    def test_required_validation_accepts_meaningful_profile(self):
        errors = validate_inputs("Bachelor's", "Python, SQL", "Data science")
        self.assertEqual(errors, [])

    def test_confidence_band_requires_profile_alignment(self):
        self.assertEqual(confidence_band(0.5, 0.5, 0.0), "Exploratory match")
        self.assertEqual(confidence_band(0.42, 0.2, 0.4), "Strong recommendation")
        self.assertEqual(confidence_band(0.28, 0.08, 0.3), "Reliable ranked match")


if __name__ == "__main__":
    unittest.main()
