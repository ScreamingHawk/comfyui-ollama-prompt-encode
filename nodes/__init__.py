"""
@author: Michael Standen
@title: Ollama Prompt Encode
@nickname: Ollama Prompt Encode
@description: Use AI to generate prompts and perform CLIP text encoding
"""

from .OllamaPromptGenerator import OllamaPromptGenerator
from .OllamaClipTextEncode import OllamaCLIPTextEncode

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptGenerator": "Ollama Prompt Generator",
    "OllamaCLIPTextEncode": "Ollama CLIP Prompt Encode",
}

NODE_CLASS_MAPPINGS = {
    "OllamaPromptGenerator": OllamaPromptGenerator,
    "OllamaCLIPTextEncode": OllamaCLIPTextEncode,
}
