"""Record what each checkpoint does when an argument is left out.

Covers the two calls the pipelines make: ``apply_chat_template`` and ``generate``. Both
take their defaults from two places, so both layers are reported for each:

* layer 1 - the Python function (``ProcessorMixin.apply_chat_template``,
  ``GenerationMixin.generate``). Its named parameters carry defaults written into the
  function definition. This layer is shared by every checkpoint, so it is printed once
  per section rather than repeated per model.
* layer 2 - the files shipped with the checkpoint: the Jinja chat template, and
  ``generation_config.json``. Arguments such as ``enable_thinking``, ``do_sample`` and
  ``temperature`` are not named parameters at all; they arrive through ``**kwargs`` and
  are resolved against these files. This layer is what actually differs between
  checkpoints.

For the chat template, layer 2 is reported twice: the declaring lines as written in the
template, and an empirical check that renders the same conversation with the argument
left out and with it forced both ways, to see which one the omitted case matches.

Loads processors and configs only (no weights, no GPU) and writes to
``chat_template_defaults.txt``.

    python validate_defaults.py
"""

import inspect
import re

from transformers import AutoProcessor, GenerationConfig
from transformers.processing_utils import ProcessorMixin

try:
    from transformers import GenerationMixin
except ImportError:  # older layouts export it from the submodule
    from transformers.generation import GenerationMixin

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

# a minimal text-only conversation, rendered to see what a template argument actually changes
probe_conversation = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

# the generation fields worth reading first: they decide whether decoding stays deterministic
decoding_fields = [
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "num_beams",
    "repetition_penalty",
    "max_new_tokens",
    "max_length",
]


# --------------------------------------------------------------------------------------
# Writing helpers.
# --------------------------------------------------------------------------------------

def write_title(f, text):
    f.write("\n" + "#" * 100 + "\n")
    f.write(f"# {text}\n")
    f.write("#" * 100 + "\n\n")


def write_subtitle(f, text):
    f.write("=" * 100 + "\n")
    f.write(text + "\n")
    f.write("=" * 100 + "\n")


def write_signature(f, function):
    """Layer 1: the defaults written into the Python function itself."""
    for name, parameter in inspect.signature(function).parameters.items():
        if name == "self":
            continue
        if parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
            # a bucket for extra arguments; it has no default of its own
            value = "(bucket: collects the arguments resolved per checkpoint below)"
        elif parameter.default is inspect.Parameter.empty:
            value = "<required>"
        else:
            value = repr(parameter.default)
        f.write(f"    {name:<32} = {value}\n")


# --------------------------------------------------------------------------------------
# Chat template.
# --------------------------------------------------------------------------------------

def get_chat_template(processor):
    """Return the checkpoint's Jinja template source as a single string."""
    template = getattr(processor, "chat_template", None)
    if template is None:
        template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    # a checkpoint may ship several named templates
    if isinstance(template, dict):
        template = "\n".join(f"### template: {name}\n{text}" for name, text in template.items())
    return template or ""


def template_arguments(template):
    """Names the template reads out of ``**kwargs``, e.g. ``enable_thinking``.

    A template asks for an optional argument either as ``x is defined`` or as
    ``x | default(...)``, so both spellings are collected.
    """
    names = re.findall(r"(\w+)\s+is\s+(?:not\s+)?defined", template)
    names += re.findall(r"(\w+)\s*\|\s*default\(", template)
    return sorted(set(names))


def render(processor, **kwargs):
    """Render the probe conversation, or return None if the template refuses it."""
    try:
        return processor.apply_chat_template(
            probe_conversation, add_generation_prompt=True, tokenize=False, **kwargs
        )
    except Exception:
        return None


def effective_default(processor, name):
    """Report which value the template falls back on when ``name`` is not passed."""
    omitted = render(processor)
    as_true = render(processor, **{name: True})
    as_false = render(processor, **{name: False})

    if omitted is None:
        return "could not render the probe conversation"
    if as_true is None or as_false is None:
        return "not a true/false argument, or the template rejected it"
    if as_true == as_false:
        return "no visible effect on this conversation (True and False render the same)"
    if omitted == as_true:
        return "True"
    if omitted == as_false:
        return "False"
    return "neither (omitting it renders differently from both True and False)"


def write_chat_template_defaults(f, model_id):
    """Layer 2 for ``apply_chat_template``: what this checkpoint's own template decides."""
    try:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    except Exception as error:
        f.write(f"processor did not load: {type(error).__name__}: {error}\n\n")
        return

    template = get_chat_template(processor)
    f.write(f"processor class: {type(processor).__name__}\n")
    f.write(f"chat template: {len(template)} chars\n\n")

    # the default as the template declares it
    declarations = [
        line.strip()
        for line in template.splitlines()
        if "is defined" in line or "default(" in line or re.search(r"{%-?\s*set\s", line)
    ]
    f.write("  lines that declare a default:\n")
    for line in dict.fromkeys(declarations) or ["(none found)"]:
        f.write(f"      {line}\n")

    # the default as the template actually behaves
    arguments = template_arguments(template)
    f.write("\n  what happens when the argument is omitted:\n")
    for name in arguments:
        f.write(f"      {name:<24} -> {effective_default(processor, name)}\n")
    if not arguments:
        f.write("      (the template reads no optional arguments)\n")

    # shows, among other things, whether an opening think tag is pre-filled into the
    # prompt (which is why a decoded generation can hold only the closing tag)
    f.write("\n  prompt rendered with nothing declared:\n")
    f.write(f"      {render(processor)!r}\n\n")


# --------------------------------------------------------------------------------------
# Generate.
# --------------------------------------------------------------------------------------

def write_generate_defaults(f, model_id):
    """Layer 2 for ``generate``: what this checkpoint's generation_config.json decides.

    These are the values ``generate`` falls back on for every argument the caller does
    not pass, which is exactly what the pipelines' explicit GENERATION_ARGS override.
    """
    try:
        generation_config = GenerationConfig.from_pretrained(model_id)
    except Exception as error:
        f.write(f"generation config did not load: {type(error).__name__}: {error}\n\n")
        return

    config_dict = generation_config.to_dict()

    f.write("  decoding fields:\n")
    for name in decoding_fields:
        value = config_dict.get(name, "(not set: falls back to the transformers default)")
        f.write(f"      {name:<24} = {value!r}\n")

    # everything this checkpoint changed away from the plain transformers defaults
    f.write("\n  every field this checkpoint sets differently from the transformers default:\n")
    differences = generation_config.to_diff_dict()
    differences.pop("transformers_version", None)
    for name in sorted(differences):
        f.write(f"      {name:<24} = {differences[name]!r}\n")
    if not differences:
        f.write("      (none: the checkpoint ships the plain transformers defaults)\n")
    f.write("\n")


# --------------------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------------------

with open(output_path, "w", encoding="utf-8") as f:
    write_title(f, "CHAT TEMPLATE")
    f.write("python signature, shared by every checkpoint (ProcessorMixin.apply_chat_template):\n")
    write_signature(f, ProcessorMixin.apply_chat_template)
    f.write("\n")
    for model_id in model_checkpoints:
        write_subtitle(f, model_id)
        write_chat_template_defaults(f, model_id)

    write_title(f, "GENERATE")
    f.write("python signature, shared by every checkpoint (GenerationMixin.generate):\n")
    write_signature(f, GenerationMixin.generate)
    f.write("\n")
    for model_id in model_checkpoints:
        write_subtitle(f, model_id)
        write_generate_defaults(f, model_id)

print(f"written to {output_path}")
