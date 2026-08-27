import transformers
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import sys
from config import resolve_path, validate_image_paths

from ..generators import (AVAILABLE_MODELS, get_model_spec, load_generator, generate_prediction,
                          resolve_thinking, thinking_models, validate_thinking_request)
import time
from datetime import datetime

_T0 = time.perf_counter()
_last = _T0

def now_str():
    """Wall-clock date and time, so a log line can be matched to the SLURM job."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def stamp(label):
    """Print the wall-clock time, the time this stage took, and the time since the run began."""
    global _last
    now = time.perf_counter()
    print(f"[TIME] {now_str()} | {label} | stage took {now-_last:7.1f}s "
          f"| elapsed since start {now-_T0:7.1f}s", flush=True)
    _last = now

print(f"[TIME] {now_str()} | pipeline started - this is the wall-clock date and time the run "
      f"began; every [TIME] line below is measured from this moment", flush=True)

parser = argparse.ArgumentParser(description="Run Retrieval-Augmented Generation.")
parser.add_argument("--test_path", type=str, required=True,
                    help="the path to the text csv (the csv that needs to be classified).")
parser.add_argument("--results_path", type=str, required=True,
                    help="path to save results csv (including the name of the csv file).")
parser.add_argument("--generator_id", type=str, default="llava-hf/llava-v1.6-34b-hf",
                    choices=AVAILABLE_MODELS,
                    help="The path to the Hugging Face generator model checkpoint.")
parser.add_argument("--enable_thinking", action="store_true",
                    help="Let the generator reason before answering, and save that reasoning to a "
                         "'thinking' column. Omitting the flag means no thinking. Only checkpoints "
                         f"whose chat template takes an enable_thinking argument support it: "
                         f"{', '.join(thinking_models('optional'))}.")

args = parser.parse_args()
test_path = args.test_path
results_path = args.results_path
generator_id = args.generator_id
enable_thinking = args.enable_thinking

print("Code running. CLI call:")
for _k, _v in vars(args).items():
    print(f"  {_k}: {_v}")

generator_spec = get_model_spec(generator_id)

print("Package versions:")
print(f"versions | torch {torch.__version__} | transformers {transformers.__version__} | "
      f"numpy {np.__version__}")

print("GPU device:")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"the device being used: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# validate thinking mode request - checks whether generator model checkpoint allows. otherwise, raise an error 
validate_thinking_request(generator_id, generator_spec, enable_thinking)

# if thinking is on, this run will record "thinking" generations in results file 
thinking_on = resolve_thinking(generator_spec, enable_thinking)
print(f"the run produces thinking text: {thinking_on}")

# print conversation
def print_conversation(conv):
    for msg in conv:
        print(f"\nROLE: {msg.get('role')}")
        content = msg.get("content")

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    print("  <IMAGE>")
                elif isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    print("  TEXT:", text)
                else:
                    print("  OTHER:", item)
        else:
            print("  CONTENT:", content)

# read csv
test_df = pd.read_csv(test_path).reset_index(drop=True)
# classes list
classes_list = sorted(test_df['true_label'].unique().tolist())

# validate image paths
validate_image_paths(test_df["file_path"].tolist(), test_path)

# load the generator
generator_model, generator_processor, generator_spec = load_generator(generator_id)
stamp(f"generator loaded: {generator_id} -> Time passed for loading model")
# check the checkpoint loaded as asked: the classes the Auto loader resolved to, float16
# weights, layers spread over the GPUs and not offloaded to CPU/disk. The classes are printed
# rather than hard-coded in the loader, so the run log records the architecture that actually
# ran. The tag names the function these values come from, so it maps onto a logger name later
# (see docs/use_logging_recommendation.md).
print(f"[generators.registry.load_generator] generator classes: "
      f"{type(generator_model).__name__} / {type(generator_processor).__name__}")
print(f"[generators.registry.load_generator] generator model dtype: {next(generator_model.parameters()).dtype}")
print(f"[generators.registry.load_generator] generator device map: {generator_model.hf_device_map}")
print(f"[generators.registry.load_generator] generator quantization: "
      f"{getattr(generator_model.config, 'quantization_config', None)}")
print(f"[generators.registry.load_generator] checkpoint revision: "
      f"{getattr(generator_model.config, '_commit_hash', None)}")


from collections import Counter
vision_layers = Counter(
    type(m).__name__
    for n, m in generator_model.named_modules()
    if "vision" in n and hasattr(m, "weight")
)
print(f"[generators.registry.load_generator] vision tower layers: {dict(vision_layers)}")


# if results path exist, load it
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path).reset_index(drop=True)

    if len(results_df) != len(test_df):
        raise ValueError("results and test have different number of rows.")

    # if file path isn't exist, or file paths order mismatch
    if "file_path" not in results_df.columns or "file_path" not in test_df.columns:
        raise ValueError("Missing 'file_path' column in test_path or results_path.")

    same_order = results_df["file_path"].astype(str).equals(test_df["file_path"].astype(str))
    if not same_order:
        raise ValueError(f"file paths are not in the same order, in paths: [{results_path}],[{test_path}]")

    # create start row
    pred = results_df["prediction"]
    missing = pred.isna() | (pred.astype(str).str.strip() == "") | (pred.astype(str).str.lower() == "none")
    start_row = int(results_df.index[missing][0]) if missing.any() else len(results_df)

    if start_row >= len(test_df):
        print("All predictions already exist.")
        sys.exit(0)
else:
    results_df = test_df.copy(deep=True)
    results_df["prediction"] = None
    results_df["query_file_path"] = None
    # the thinking column exists only for a run that produces reasoning; prediction is
    # unaffected either way
    if thinking_on:
        results_df["thinking"] = None
    start_row = 0

print_debug = True
batch_size = 100
# how many samples this run classifies, and how many batches that takes. keep in mind the
# last batch might not contain "batch size" samples
num_samples = len(test_df) - start_row
num_batches = (num_samples + batch_size - 1) // batch_size

check_output_truncation = True
truncated_count = 0
for curr_batch, batch_start in enumerate(range(start_row, len(test_df), batch_size)):
    batch_end = min(batch_start + batch_size, len(test_df))
    batch_df = test_df.iloc[batch_start:batch_end].copy()
    batch_predictions = []
    batch_thinking = []
    query_file_paths = []
    # memory usage
    torch.cuda.reset_peak_memory_stats()
    # loop through images to classify
    for _, row in batch_df.iterrows():
        # load query
        query_path = row["file_path"]
        query_file_paths.append(query_path)
        absolute_query_path = resolve_path(query_path)
        with Image.open(absolute_query_path) as im:
            query_image = im.convert("RGB")


        conversation = []
        # system role
        conversation.append({
            "role": "system",
            "content": [
                {"type": "text",
                 "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                         f"You are given a query image. Analyze the facial expression in the query image and classify the emotion.\n"
                         f"Follow the user's requested output format."
                 }
            ]
        })

        # user role
        conversation.append({"role": "user",
                             "content": [{
                                 "type": "image"},
                                 {"type": "text", "text": f"Classify the emotion shown in this image into one of the following emotions: {', '.join(classes_list)}.\n"
                                                          f"Respond with only one word: the emotion label."}]})

        # process inputs
        if print_debug == True:
            print("conversation:")
            print_conversation(conversation)
            print(f"conversation structure:\n{conversation}")
 
        # generate prediction
        prediction, thinking, gen_stats = generate_prediction(generator_model, generator_processor, conversation,
                                                              [query_image], generator_spec,
                                                              enable_thinking=enable_thinking,
                                                              debug=print_debug,
                                                              check_truncation=check_output_truncation)

        # the first sample is dumped in full: the prompt the model is actually given (the
        # conversation above is what was sent, this is what it became), the tensors it
        # receives, and the raw generation next to the thinking and prediction that
        # decode_generation split out of it. Printed with !r so that an empty string stays
        # distinguishable from None.
        if print_debug == True:
            debug_info = gen_stats.pop("debug")
            print(f"gen_stats: {gen_stats}")
            print(f"input tensors the model receives: {debug_info['input_keys']}")
            print(f"prompt the model actually sees:\n{debug_info['prompt_text']}")
            print(f"raw generation, special tokens kept: {debug_info['raw_generated']!r}")
            print(f'prediction: {prediction!r}')
            print(f'thinking: {thinking!r}')
            print_debug = False

        # generation that ran out of budget (max new tokens) rather than finishing its answer. 
        if gen_stats["finish_reason"] == "length":
            truncated_count += 1
            if truncated_count == 1:
                print(f"[WARNING] pipelines.zero_shot: the output was truncated by max_new_tokens="
                      f"{gen_stats['max_new_tokens']}, so the answer may be incomplete. Consider "
                      f"raising max_new_tokens for '{generator_id}'."
                      f"prediction: {prediction!r}, query: {query_path}")
        elif gen_stats["finish_reason"] == "unknown":
            print(f"[WARNING] pipelines.zero_shot: '{generator_id}' does not declare an end of "
                  f"generation token, so the truncation validation cannot be performed for this "
                  f"run.")
            # the checkpoint cannot start declaring one later, so there is nothing more to learn
            check_output_truncation = False

        batch_predictions.append(prediction)
        batch_thinking.append(thinking)

        del query_image
        torch.cuda.empty_cache()
    # end batch

    # validate predictions are not empty or none
    empty_count = sum(1 for p in batch_predictions if not p)
    if empty_count:
        print(f"[WARNING] {empty_count} empty predictions in this batch - generation ended "
              f"before an answer. Consider raising max_new_tokens for '{generator_id}'.")

    # update results
    results_df.loc[batch_start:batch_end - 1, "prediction"] = batch_predictions
    results_df.loc[batch_start:batch_end - 1, "query_file_path"] = query_file_paths
    # the reasoning that produced those predictions, on the same rows
    if thinking_on:
        results_df.loc[batch_start:batch_end - 1, "thinking"] = batch_thinking
    results_df.to_csv(results_path, index=False)
    print(f"peak GPU memory this batch: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    torch.cuda.empty_cache()
    # end of batch 
    what_batch = curr_batch + 1
    stamp(f"batch {what_batch}/{num_batches} done, ({len(batch_predictions)} samples) -> Time for "
          f"processing the batch")

# a few truncated samples are noise, a large share means the generation budget is too small
# for this configuration. The rate is what tells the two apart.
if truncated_count:
    print(f"[WARNING] pipelines.zero_shot: {truncated_count} of {num_samples} samples were "
          f"truncated before the model finished its answer. Consider raising max_new_tokens for "
          f"'{generator_id}'.")

print(f"[TIME] {now_str()} | pipeline ended -> total runtime "
      f"{(time.perf_counter()-_T0)/60:.1f} min", flush=True)
