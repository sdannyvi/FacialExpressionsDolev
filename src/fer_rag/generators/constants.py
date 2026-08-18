"""Shared decoding defaults for the generator layer.

These are the settings that apply to every checkpoint. Per-model overrides (currently
only ``max_new_tokens``) live in the registry entry in ``registry.py``.
"""
import torch
from transformers import BitsAndBytesConfig
# Generation arguments shared by every generator. These are passed explicitly to every
# model rather than relying on library defaults, so that decoding stays deterministic
# even if a future transformers release changes its defaults. Per-model overrides live
# in the registry (currently only max_new_tokens).
GENERATION_ARGS = {
    "do_sample": False,
    "temperature": 1.0,
    "num_beams": 1,
}

# Fallback when a registry entry does not override it.
DEFAULT_MAX_NEW_TOKENS = 20

# configure quantization to load models 
QUANTIZATION = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          
    bnb_4bit_compute_dtype=torch.bfloat16,  
    bnb_4bit_use_double_quant=True,    
)