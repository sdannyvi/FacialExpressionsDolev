"""Record what ``apply_chat_template`` accepts, and what it defaults to, per checkpoint.

Loads the processor only (no weights, no GPU) and writes the argument list of each
checkpoint's ``apply_chat_template`` to ``chat_template_defaults.txt``, one section per
model. Standalone on purpose: it imports nothing from ``src/``, so it can be run on its own.

    python validate_defaults.py
"""

import inspect

from transformers import AutoProcessor

# the checkpoints registered in src/fer_rag/generators/registry.py
model_checkpoints = [
    "llava-hf/llava-v1.6-34b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "google/gemma-3-27b-it",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
]

output_path = "chat_template_defaults.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for model_id in model_checkpoints:
        # title
        f.write("=" * 100 + "\n")
        f.write(model_id + "\n")
        f.write("=" * 100 + "\n")

        # a gated or missing repo should not abort the remaining checkpoints
        try:
            processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        except Exception as error:
            f.write(f"processor did not load: {type(error).__name__}: {error}\n\n")
            continue

        f.write(f"processor class: {type(processor).__name__}\n")

        # the arguments and their defaults, read straight off the bound method
        method = processor.apply_chat_template
        f.write(f"apply_chat_template defined on: {getattr(method, '__qualname__', '?')}\n")
        for name, parameter in inspect.signature(method).parameters.items():
            default = "<required>" if parameter.default is inspect.Parameter.empty else repr(parameter.default)
            f.write(f"    {name:<28} = {default:<30} ({parameter.kind.name.lower()})\n")
        f.write("\n")

print(f"written to {output_path}")
