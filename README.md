# ComfyUI Ollama Prompt Encode

A prompt generator and CLIP encoder using AI provided by [Ollama](https://ollama.com).

## Prerequisites

Install [Ollama](https://ollama.com) and have the service running.

## Usage

![Example Usage](./docs/usage_1.png)

The `Ollama CLIP Prompt Encode` node is designed to replace the default `CLIP Text Encode (Prompt)` node. It generates a prompt using the Ollama AI model and then encodes the prompt with CLIP.

The node will output the generated prompt as a `string`. This can be viewed with [rgthree's `Display Any` node](https://github.com/rgthree/rgthree-comfy?tab=readme-ov-file#display-any).

### Ollama URL

The URL to the Ollama service. The default is `http://localhost:11434`.

### Ollama Model

This is the model that is used to generate your prompt.

Some models that work well with this prompt generator are:

- `orca-mini`
- `mistral`
- `tinyllama`

### Prepend Tags

A string that will be prepended to the generated prompt.

This is useful for models like `pony` that work best with extra tags like `score_9, score_8_up`.

### Text

The text that will be used by the AI model to generate the prompt.

## Testing

Run the tests with:

```sh
python -m unittest
```
