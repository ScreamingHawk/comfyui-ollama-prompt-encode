"""
@author: Michael Standen
@title: Ollama Prompt Generator
@nickname: Ollama Prompt Generator
@description: Use AI to generate prompts
"""

import os
import csv
from ollama import Client, Options
from .timeout import timeout

SYSTEM_MESSAGES = {
    "descriptive": "You describe pictures. I will give you a brief description of the picture. You will describe the picture in intricate detail. Describe clothing, pose, expression, setting, lighting, and any other details you can think of. Use long descriptive sentences.",
    "comma": "You describe pictures. I will give you a brief description of the picture. You reply with comma separated keywords that describe the picture. Describe clothing, pose, expression, setting, and any other details you can think of. Use comma separated keywords. Do not use sentences. Use brevity.",
}

class OllamaPromptGenerator:
    # Defaults
    OLLAMA_TIMEOUT = 60
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "orca-mini"

    def load_sample_data(self, comma_separated_response: bool = True):
        fname = "sample_data_comma.csv" if comma_separated_response else "sample_data_descriptive.csv"
        fname = os.path.join(os.path.dirname(__file__), fname)
        sample_data = []
        with open(fname, "r") as fin:
            reader = csv.DictReader(fin)
            sample_data = [row for row in reader]
        return sample_data

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "ollama_model": ("STRING", {"default": cls.OLLAMA_MODEL}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prepend_tags": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "comma_separated_response": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "prompt",
    )
    FUNCTION = "get_prompt"

    CATEGORY = "Ollama"

    def sanitize_prompt(self, prompt):
        """Sanitize the prompt for use in clip encoding."""
        return prompt.replace(".", ",")

    @timeout(OLLAMA_TIMEOUT)
    def generate_prompt(self, ollama_url, ollama_model, text, seed: int|None = None, comma_separated_response: bool = True):
        """Get a prompt from the Ollama API."""
        ollama_client = Client(host=ollama_url)

        # Download the model if it doesn't exist
        ollama_client.pull(ollama_model)

        opts = Options()
        if seed is not None:
            opts["seed"] = seed
            opts["temperature"] = 0.0

        # System message
        system_message = SYSTEM_MESSAGES["comma"] if comma_separated_response else SYSTEM_MESSAGES["descriptive"]
        messages = [
            {"role": "system", "content": system_message},
        ]

        # Sample data
        sample_data = self.load_sample_data(comma_separated_response)
        for row in sample_data:
            messages.append({"role": "user", "content": "Write a prompt for: " + row["text"]})
            messages.append({"role": "assistant", "content": row["prompt"]})

        # User prompt
        messages.append({"role": "user", "content": "Write a prompt for: " + text})

        response = ollama_client.chat(
            model=ollama_model,
            messages=messages,
            options=opts,
            stream=False,
        )

        prompt = response["message"]["content"]

        return prompt

    def get_prompt(self, ollama_url, ollama_model, seed, prepend_tags, text, comma_separated_response):
        """Generates prompt using Ollama."""
        use_seed = seed if seed != 0 else None
        prompt = self.generate_prompt(ollama_url, ollama_model, text, use_seed, comma_separated_response)
        combined_prompt = prepend_tags + ", " + self.sanitize_prompt(prompt)

        return (combined_prompt,)
