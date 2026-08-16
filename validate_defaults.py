"""Record the full set of values each checkpoint uses when the caller passes nothing.

The question this answers is: *if I call ``apply_chat_template(conversation)`` or
``generate(**inputs)`` and specify no other argument, what value does every single
argument actually take for this checkpoint?*

Nothing is filtered and no argument is skipped. Each call is reported as the layers that
together decide a value:

* the **named parameters** of the Python function, whose defaults are written into the
  function definition and are the same for every checkpoint;
* the **checkpoint's own files** - the Jinja chat template for ``apply_chat_template``,
  and ``generation_config.json`` for ``generate``. Arguments such as ``enable_thinking``
  or ``temperature`` are not named parameters at all: they arrive through ``**kwargs``
  and are resolved against these files. This layer is what differs between checkpoints;
* the **library fallback** for everything neither of the above sets.

For ``generate`` this means every field of ``GenerationConfig`` is listed, each tagged
with where its value came from, so an argument that the checkpoint leaves alone is still
reported with the value it will really have.

Loads processors and configs only (no weights, no GPU) and writes to
``chat_template_defaults.txt``.

    python validate_defaults.py
"""

import inspect
import re

from transformers import AutoProcessor, GenerationConfig, logging
from transformers.processing_utils import ProcessorMixin

try:
    from transformers import GenerationMixin
except ImportError:  # older layouts export it from the submodule
    from transformers.generation import GenerationMixin

try:
    # the helper transformers itself uses to split template kwargs from processor kwargs
    from transformers.processing_utils import _get_template_variables
except ImportError:  # older versions have no such helper, fall back to reading the source
    _get_template_variables = None

logging.set_verbosity_error()  # keep the report clean of load-time chatter

# the last layer generate consults: values applied to whatever is still None after the
# checkpoint's generation_config.json has been read
try:
    global_generation_defaults = GenerationConfig._get_default_generation_params()
except AttributeError:  # older versions bake these into GenerationConfig.__init__ instead
    global_generation_defaults = {}

# names that appear as recommended settings on model cards but are not transformers
# arguments; GenerationConfig stores them as unused attributes, so passing them is silent
model_card_aliases = {
    "out_seq_length": "not a transformers argument; the equivalent is max_new_tokens",
    "presence_penalty": "vLLM/SGLang only; transformers has repetition_penalty instead",
    "frequency_penalty": "vLLM/SGLang only; transformers has repetition_penalty instead",
    "greedy": "not a transformers argument; the equivalent is do_sample=False",
    "max_tokens": "not a transformers argument; the equivalent is max_new_tokens",
}

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

# names a template sees without them ever being passed by the caller
injected_by_jinja = {"raise_exception", "strftime_now"}
supplied_conversation = {"messages"}


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


def signature_defaults(function):
    """The defaults written into the Python function definition itself."""
    defaults = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name == "self":
            continue
        if parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
            defaults[name] = "(bucket: collects the arguments resolved per checkpoint below)"
        elif parameter.default is inspect.Parameter.empty:
            defaults[name] = "<required: supplied by the caller>"
        else:
            defaults[name] = repr(parameter.default)
    return defaults


# --------------------------------------------------------------------------------------
# Chat template.
# --------------------------------------------------------------------------------------

def get_chat_template(processor):
    """Return the checkpoint's Jinja template source as a single string."""
    template = getattr(processor, "chat_template", None)
    if template is None:
        template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    # a checkpoint may ship several named templates; apply_chat_template uses "default"
    if isinstance(template, dict):
        template = template.get("default") or "\n".join(template.values())
    return template or ""


def template_variables(template):
    """Every name the template reads but never declares itself.

    Uses the same ``jinja2.meta.find_undeclared_variables`` call that
    ``ProcessorMixin.apply_chat_template`` makes, so the list matches exactly what
    transformers treats as a template argument.
    """
    if _get_template_variables is not None:
        return sorted(_get_template_variables(template))
    names = re.findall(r"(\w+)\s+is\s+(?:not\s+)?defined", template)
    names += re.findall(r"(\w+)\s*\|\s*default\(", template)
    return sorted(set(names))


def declared_default(template, name):
    """The literal a template writes for a name, as in ``x | default(false)``."""
    match = re.search(rf"\b{re.escape(name)}\s*\|\s*default\(\s*([^),]*)", template)
    return match.group(1).strip() if match else None


def render(processor, **kwargs):
    """Render the probe conversation, or return None if the template refuses it."""
    try:
        return processor.apply_chat_template(probe_conversation, tokenize=False, **kwargs)
    except Exception:
        return None


def observed_behaviour(processor, name):
    """Which value the omitted case matches, checked by rendering rather than reading."""
    omitted = render(processor)
    as_true = render(processor, **{name: True})
    as_false = render(processor, **{name: False})

    if omitted is None:
        return "could not render the probe conversation"
    if as_true is None or as_false is None:
        return "not a true/false argument"
    if as_true == as_false:
        return "no visible effect on a plain text conversation"
    if omitted == as_true:
        return "omitting it behaves like True"
    if omitted == as_false:
        return "omitting it behaves like False"
    return "omitting it renders unlike both True and False"


def write_chat_template_defaults(f, model_id, named_defaults):
    """Every value in play when only ``conversation`` is passed to this checkpoint."""
    try:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    except Exception as error:
        f.write(f"processor did not load: {type(error).__name__}: {error}\n\n")
        return

    template = get_chat_template(processor)
    f.write(f"processor class: {type(processor).__name__}\n")
    f.write(f"chat template: {len(template)} chars\n\n")

    # ---- layer 1: the named parameters of the Python function
    f.write("  [1] named parameters of ProcessorMixin.apply_chat_template\n")
    for name, value in named_defaults.items():
        note = ""
        if name == "chat_template":
            note = "-> falls back to this checkpoint's own template"
        elif name == "tokenize":
            note = "-> no tokenizing, so no processor kwargs are consulted"
        f.write(f"      {name:<34} = {value:<44}{(' ' + note) if note else ''}\n")

    # ---- layer 2: everything the checkpoint's own template reads
    variables = template_variables(template)
    f.write("\n  [2] variables this checkpoint's Jinja template reads, and their value when nothing is passed\n")
    for name in variables:
        if name in injected_by_jinja:
            value, note = "<jinja helper>", "supplied by the transformers template environment"
        elif name in supplied_conversation:
            value, note = "<the conversation>", "the argument you pass"
        elif name in named_defaults:
            value, note = named_defaults[name], "named parameter above"
        else:
            literal = declared_default(template, name)
            if literal is not None:
                value, note = literal, f"template kwarg, default() in the template; {observed_behaviour(processor, name)}"
            else:
                value, note = "<undefined>", f"template kwarg, never defaulted so Jinja leaves it undefined (falsy); {observed_behaviour(processor, name)}"
        f.write(f"      {name:<34} = {value:<44}{(' ' + note) if note else ''}\n")
    if not variables:
        f.write("      (the template reads no variables at all)\n")

    # ---- layer 3: the processor kwargs that would apply if tokenize were True
    class_defaults = getattr(getattr(processor, "valid_processor_kwargs", None), "_defaults", None)
    f.write("\n  [3] processor kwargs defaults on this class (only reached when tokenize=True)\n")
    if class_defaults:
        for group in sorted(class_defaults):
            f.write(f"      {group}:\n")
            for name in sorted(class_defaults[group]):
                f.write(f"          {name:<30} = {class_defaults[group][name]!r}\n")
    else:
        f.write("      (this processor class declares no kwarg defaults of its own)\n")

    # ---- the resulting prompt, which shows e.g. a pre-filled opening think tag
    f.write("\n  [4] prompt produced when only the conversation is passed\n")
    f.write(f"      {render(processor)!r}\n")
    f.write("      with add_generation_prompt=True (what the pipelines pass):\n")
    f.write(f"      {render(processor, add_generation_prompt=True)!r}\n\n")


# --------------------------------------------------------------------------------------
# Generate.
# --------------------------------------------------------------------------------------

def write_generate_defaults(f, model_id, named_defaults):
    """Every value in play when ``generate`` is called with inputs and nothing else.

    ``generate`` fills an unpassed argument in a fixed order (see
    ``GenerationMixin._prepare_generation_config``): the checkpoint's
    ``generation_config.json`` first, then ``_get_default_generation_params()`` for
    whatever is still ``None``. A ``None`` in the config file is therefore not the value
    that gets used - it only means "not set here, ask the next layer" - so every field is
    reported already resolved, together with the layer it came from.
    """
    try:
        generation_config = GenerationConfig.from_pretrained(model_id)
    except Exception as error:
        f.write(f"generation config did not load: {type(error).__name__}: {error}\n\n")
        return

    config_dict = generation_config.to_dict()
    from_checkpoint = set(generation_config.to_diff_dict()) - {"transformers_version"}

    resolved = {}
    for name in sorted(set(config_dict) | set(global_generation_defaults)):
        value = config_dict.get(name)
        if name in from_checkpoint:
            resolved[name] = (value, "generation_config.json")
        elif value is not None:
            resolved[name] = (value, "library default")
        elif name in global_generation_defaults:
            resolved[name] = (global_generation_defaults[name], "transformers global default")
        else:
            resolved[name] = (None, "never set: stays None")

    # ---- layer 1: the named parameters of generate itself
    f.write("  [1] named parameters of GenerationMixin.generate\n")
    for name, value in named_defaults.items():
        note = "-> falls back to the model's generation config, resolved below" if name == "generation_config" else ""
        f.write(f"      {name:<34} = {value:<44}{(' ' + note) if note else ''}\n")

    # ---- layer 2: every generation argument, resolved, with where its value came from
    f.write(f"\n  [2] every generation argument ({len(resolved)} of them), with the value actually used\n")
    for name, (value, source) in resolved.items():
        note = ""
        if name == "max_new_tokens" and value is None:
            note = f"-> so the cap comes from max_length={resolved['max_length'][0]!r} instead"
        f.write(f"      {name:<34} = {value!r:<40}[{source}]{(' ' + note) if note else ''}\n")

    # ---- the short version: what decoding will actually do
    def used(name):
        return resolved[name][0]

    f.write("\n  [3] what this means when generate is called with inputs only\n")
    if used("do_sample"):
        f.write(
            f"      sampling on: temperature={used('temperature')!r}, top_p={used('top_p')!r}, "
            f"top_k={used('top_k')!r}, repetition_penalty={used('repetition_penalty')!r}\n"
        )
    else:
        f.write(f"      greedy decoding (do_sample={used('do_sample')!r}); temperature/top_p/top_k are ignored\n")
    # max_length means different things depending on which layer supplied it: a value the
    # checkpoint sets caps prompt+output together, whereas the untouched global default is
    # treated as "20 new tokens" because generate adds the prompt length back onto it
    if used("max_new_tokens") is not None:
        f.write(f"      length: max_new_tokens={used('max_new_tokens')!r}, so {used('max_new_tokens')!r} new tokens\n")
    elif resolved["max_length"][1] == "generation_config.json":
        f.write(
            f"      length: max_new_tokens is unset, so max_length={used('max_length')!r} caps prompt and output\n"
            f"              together - the prompt eats into what is left to generate\n"
        )
    else:
        f.write(
            f"      length: neither max_new_tokens nor max_length is set anywhere, so generate falls back to the\n"
            f"              model-agnostic max_length={used('max_length')!r} and adds the prompt length back onto it,\n"
            f"              which yields exactly {used('max_length')!r} new tokens (and warns that you should set max_new_tokens)\n"
        )
    f.write("\n")


# --------------------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------------------

chat_template_named = signature_defaults(ProcessorMixin.apply_chat_template)
generate_named = signature_defaults(GenerationMixin.generate)

with open(output_path, "w", encoding="utf-8") as f:
    write_title(f, "APPLY_CHAT_TEMPLATE")
    for model_id in model_checkpoints:
        write_subtitle(f, model_id)
        write_chat_template_defaults(f, model_id, chat_template_named)

    write_title(f, "GENERATE")
    f.write("names used by model cards that are NOT transformers arguments\n")
    f.write("(GenerationConfig keeps them as unused attributes, so passing them changes nothing):\n")
    for name, meaning in model_card_aliases.items():
        exists = hasattr(GenerationConfig(), name)
        f.write(f"    {name:<24} present in GenerationConfig: {str(exists):<6} {meaning}\n")
    f.write("\n")
    for model_id in model_checkpoints:
        write_subtitle(f, model_id)
        write_generate_defaults(f, model_id, generate_named)

print(f"written to {output_path}")
