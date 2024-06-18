"""
@author: Michael Standen
@title: Ollama Prompt Encode
@nickname: Ollama Prompt Encode
@description: Use AI to generate prompts and perform CLIP text encoding
"""

from ollama import Client, Options
from typing import Mapping
from .timeout import timeout

class OllamaCLIPTextEncode:
    # Defaults
    OLLAMA_TIMEOUT = 30
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "orca-mini"
    OLLAMA_SYSTEM_MESSAGE = "You are a prompt engineer that generates prompts for image generation AI. The 'prompt' is short comma separated descriptors, NOT A SENTENCE. The user will describe an image, then you will respond with the 'prompt'. You will not write any text except the 'prompt'."
    OLLAMA_EXAMPLE_TEXT = "sexy sweaty cheerleader, jumping"
    OLLAMA_EXAMPLE_PROMPT = "big smile, happy, joy, light blush, jumping, dynamic pose, cheerleader, cheerleader outfit, cheerleader pom poms, thigh highs, stockings, short pleated skirt, crop top, ribbons, bow on shirt, school gym, underboob, midriff, navel, no panties, natural skin, small breast, 20 year old, warm light, dappered light, golden hour, highly detailed, detailed,"

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

    CATEGORY = "conditioning"

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

        response = ollama_client.chat(
            model=ollama_model,
            stream=False,
            messages=[
                {"role": "system", "content": self.OLLAMA_SYSTEM_MESSAGE},
                {"role": "user", "content": "Write a prompt for " + self.OLLAMA_EXAMPLE_TEXT,},
                {"role": "assistant", "content": self.OLLAMA_EXAMPLE_PROMPT},
                {"role": "user", "content": "Write a prompt for " + text},
            ],
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
