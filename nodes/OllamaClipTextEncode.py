"""
@author: Michael Standen
@title: Ollama Prompt Encode
@nickname: Ollama Prompt Encode
@description: Use AI to generate prompts and perform CLIP text encoding
"""

import os
import csv
from ollama import Client, Options
from typing import Mapping
from .timeout import timeout

# Read data from sample_data.csv
fname = os.path.join(os.path.dirname(__file__), "sample_data.csv")
sample_data = []
with open(fname, "r") as fin:
    reader = csv.DictReader(fin)
    sample_data = [row for row in reader]

class OllamaCLIPTextEncode:
    # Defaults
    OLLAMA_TIMEOUT = 30
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "orca-mini"
    OLLAMA_SYSTEM_MESSAGE = "You describe pictures. I will give you a brief description of the picture. You reply with comma separated keywords that describe the picture. Describe clothing, pose, expression, setting, and any other details you can think of. Use comma separated keywords. Do not use sentences. Use brevity."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "ollama_url": ("STRING", {"default": cls.OLLAMA_URL}),
                "ollama_model": ("STRING", {"default": cls.OLLAMA_MODEL}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prepend_tags": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    RETURN_NAMES = (
        "conditioning",
        "prompt",
    )
    FUNCTION = "get_encoded"

    CATEGORY = "Ollama"

    def sanitize_prompt(self, prompt):
        """Sanitize the prompt for use in clip encoding."""
        return prompt.replace(".", ",")

    @timeout(OLLAMA_TIMEOUT)
    def generate_prompt(self, ollama_url, ollama_model, text, seed: int|None = None):
        """Get a prompt from the Ollama API."""
        ollama_client = Client(host=ollama_url)

        # Download the model if it doesn't exist
        ollama_client.pull(ollama_model)

        opts = Options()
        if seed is not None:
            opts["seed"] = seed

        messages = [
            {"role": "system", "content": self.OLLAMA_SYSTEM_MESSAGE},
        ]
        for row in sample_data:
            messages.append({"role": "user", "content": "Write a prompt for: " + row["text"]})
            messages.append({"role": "assistant", "content": row["prompt"]})
        messages.append({"role": "user", "content": "Write a prompt for: " + text})

        response = ollama_client.chat(
            model=ollama_model,
            stream=False,
            messages=messages,
            options=opts
        )

        # Streaming not supported
        if not isinstance(response, Mapping):
            raise ValueError("Streaming not supported")

        prompt = response["message"]["content"]

        return prompt

    def get_encoded(self, clip, ollama_url, ollama_model, seed, prepend_tags, text):
        """Gets and encodes the prompt using CLIP."""
        use_seed = seed if seed != 0 else None
        prompt = self.generate_prompt(ollama_url, ollama_model, text, use_seed)
        combined_prompt = prepend_tags + ", " + self.sanitize_prompt(prompt)

        tokens = clip.tokenize(combined_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], combined_prompt)
