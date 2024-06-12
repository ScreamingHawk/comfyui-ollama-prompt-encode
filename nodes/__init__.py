"""
@author: Michael Standen
@title: Ollama Prompt Generator
@nickname: Ollama Prompt Gen
@description: Use AI to generate prompts and perform CLIP text encoding
"""

from .OllamaClipTextEncode import OllamaCLIPTextEncode

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaCLIPTextEncode": "Ollama CLIP Prompt Encode",
}

NODE_CLASS_MAPPINGS = {
    "OllamaCLIPTextEncode": OllamaCLIPTextEncode,
}
