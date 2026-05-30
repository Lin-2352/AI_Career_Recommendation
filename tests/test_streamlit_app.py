import unittest

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_app_renders_without_exceptions(self) -> None:
        app = AppTest.from_file("app.py")
        app.run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.selectbox), 3)
        self.assertGreaterEqual(len(app.text_area), 6)
        self.assertGreaterEqual(len(app.button), 5)


if __name__ == "__main__":
    unittest.main()
