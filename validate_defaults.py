from transformers import AutoProcessor, AutoTokenizer

CKPTS = [
    "llava-hf/llava-v1.6-34b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "google/gemma-3-27b-it",
    "google/gemma-4-31b-it",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
]

# a 2-example + query conversation, single user message
msgs = [
    {"role": "system", "content": [{"type": "text", "text": "You are an FER expert."}]},
    {"role": "user", "content": [
        {"type": "text",  "text": "Example 1 label: happy"},
        {"type": "image"},
        {"type": "text",  "text": "Example 2 label: sad"},
        {"type": "image"},
        {"type": "text",  "text": "Query image:"},
        {"type": "image"},
        {"type": "text",  "text": "What emotion is in the query image?"},
    ]},
]

for ckpt in CKPTS:
    print("=" * 80, "\n", ckpt)
    try:
        proc = AutoProcessor.from_pretrained(ckpt)
        tmpl = getattr(proc, "chat_template", None) or getattr(proc.tokenizer, "chat_template", None)
    except Exception:
        proc = AutoTokenizer.from_pretrained(ckpt)
        tmpl = proc.chat_template

    print("\n--- RAW TEMPLATE ---\n", tmpl)
    print("\n--- RENDERED ---\n",
          proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))