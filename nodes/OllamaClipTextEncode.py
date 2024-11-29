"""
@author: Michael Standen
@title: Ollama Prompt Encode
@nickname: Ollama Prompt Encode
@description: Use AI to generate prompts and perform CLIP text encoding
"""

from .OllamaPromptGenerator import OllamaPromptGenerator

class OllamaCLIPTextEncode(OllamaPromptGenerator):

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
                "comma_separated_response": ("BOOLEAN", {"default": True}),
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

    def get_encoded(self, clip, ollama_url, ollama_model, seed, prepend_tags, text, comma_separated_response):
        """Gets and encodes the prompt using CLIP."""
        combined_prompt = self.get_prompt(ollama_url, ollama_model, seed, prepend_tags, text, comma_separated_response)[0]

        tokens = clip.tokenize(combined_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], combined_prompt)
