import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.OllamaPromptGenerator import OllamaPromptGenerator

class TestOllamaPromptGenerator(unittest.TestCase):
    # Default values
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "orca-mini"
    SEED = 42069

    def text_sanitize_prompt(self):
        # Arrange
        encoder = OllamaPromptGenerator()
        prompt = "This is a test."

        # Act
        result = encoder.sanitize_prompt(prompt)

        # Assert
        self.assertEqual(result, "This is a test,")

    def test_generate_prompt_comma(self, retry=3):
        # Arrange
        encoder = OllamaPromptGenerator()
        text = "cute tan girl wearing nothing but overalls painting on a canvas"

        # Act
        result = encoder.generate_prompt(self.OLLAMA_URL, self.OLLAMA_MODEL, text, None, True)
        print(result)

        # Assert
        if retry > 0 and ("paint" not in result or "color" not in result):
            print("Retrying test_generate_prompt... Attempts left:", retry - 1)
            self.test_generate_prompt(retry - 1)
        # These do not always pass as this test is intentionally not using a seed
        self.assertTrue("paint" in result)
        self.assertTrue("color" in result)
        num_commas = result.count(",")
        self.assertGreater(num_commas, 5)

    def test_generate_prompt_descriptive(self, retry=3):
        # Arrange
        encoder = OllamaPromptGenerator()
        text = "cute tan girl wearing nothing but overalls painting on a canvas"

        # Act
        result = encoder.generate_prompt(self.OLLAMA_URL, self.OLLAMA_MODEL, text, None, False)
        print(result)

        # Assert
        if retry > 0 and ("paint" not in result or "color" not in result):
            print("Retrying test_generate_prompt... Attempts left:", retry - 1)
            self.test_generate_prompt_descriptive(retry - 1)
        # These do not always pass as this test is intentionally not using a seed
        self.assertTrue("paint" in result)
        self.assertTrue("color" in result)
        num_commas = result.count(",")
        self.assertLess(num_commas, 5)

    def test_generate_prompt_with_seed(self):
        # Arrange
        encoder = OllamaPromptGenerator()
        text = "princess cat on her throne"

        # Act
        res1 = encoder.generate_prompt(self.OLLAMA_URL, self.OLLAMA_MODEL, text, self.SEED, True)
        res2 = encoder.generate_prompt(self.OLLAMA_URL, self.OLLAMA_MODEL, text, self.SEED, True)
        print(res1)
        print(res2)

        # Assert
        self.assertEqual(res1, res2)

if __name__ == '__main__':
    unittest.main()
