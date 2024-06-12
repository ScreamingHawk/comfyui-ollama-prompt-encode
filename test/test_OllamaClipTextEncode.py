import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes.OllamaClipTextEncode import OllamaCLIPTextEncode

class TestOllamaCLIPTextEncode(unittest.TestCase):
    # Default values
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "orca-mini"

    def text_sanitize_prompt(self):
        # Arrange
        encoder = OllamaCLIPTextEncode()
        prompt = "This is a test."

        # Act
        result = encoder.sanitize_prompt(prompt)

        # Assert
        self.assertEqual(result, "This is a test,")

    def test_generate_prompt(self):
        # Arrange
        encoder = OllamaCLIPTextEncode()
        text = "cute tan girl, wearing nothing but overalls, painting a on a canvas"

        # Act
        result = encoder.generate_prompt(self.OLLAMA_URL, self.OLLAMA_MODEL, text)
        print(result)

        # Assert
        self.assertTrue("paint" in result)
        self.assertTrue("color" in result)

if __name__ == '__main__':
    unittest.main()
