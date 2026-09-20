import transformers
import sklearn
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
import faiss
import numpy as np
from config import resolve_path, validate_image_paths

from ..generators import (AVAILABLE_MODELS, get_model_spec, load_generator, generate_prediction,
                          resolve_thinking, thinking_models, validate_prompt_request,
                          validate_thinking_request, get_context_window)
from ..frameworks.registry import (FRAMEWORKS, framework_columns, needs_new_framework,
                                   oracle_needs_new_framework, run_framework, validate_framework_request)
from .prompts import build_rag_conversation
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

# parameters
parser = argparse.ArgumentParser(description="Run Retrieval-Augmented Generation.")
parser.add_argument('--start_batch', type=int, default=0,
                    help="Batch number to start from. if batch 0 then starting from the first row in test set. "
                         "if batch is 2 then  starting classify from row 200 in test set.")
parser.add_argument("--results_path", type=str, required=True,
                    help="path to save results csv (including the name of the csv file).")
parser.add_argument("--knowledge_base_path", type=str, required=True,
                    help="the path to the knowledge base csv file.")
parser.add_argument("--test_path", type=str, required=True,
                    help="the path to the text csv (the csv that needs to be classified).")
parser.add_argument("--generator_id", type=str, default="llava-hf/llava-v1.6-34b-hf",
                    choices=AVAILABLE_MODELS,
                    help="The path to the Hugging Face generator model checkpoint.")
parser.add_argument("--clip_model_id", type=str, default="openai/clip-vit-large-patch14",
                    choices=["openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32"],
                    help="The path to the Hugging Face CLIP model checkpoint.")
parser.add_argument("--dim_reduction", type=str, default="lda", choices=["lda", "none"],
                    help="Choose the dimensionality reduction method: 'lda' or 'none'.")
parser.add_argument("--prompt",type=str,  default="single-user-message",
                    choices = ["single-user-message","multi-user-message"],
                    help="Prompt structure.")
parser.add_argument("--top_k",type=int, default=2,
                    help="Number of top examples to retrieve from the knowledge base.")
parser.add_argument("--enable_thinking", action="store_true",
                    help="Let the generator reason before answering, and save that reasoning to a "
                         "'thinking' column. Omitting the flag means no thinking. Only checkpoints "
                         f"whose chat template takes an enable_thinking argument support it: "
                         f"{', '.join(thinking_models('optional'))}.")
parser.add_argument("--framework", type=str, default="original", choices=["original", *FRAMEWORKS],
                    help="'original' runs the original RAG on every sample. A framework name sends every "
                         "sample whose top-1 and top-2 retrieved labels differ to that framework, and the "
                         "rest to the original RAG. Frameworks are defined in frameworks/registry.py and "
                         "run with --top_k 2 on llava-hf/llava-v1.6-34b-hf.")
parser.add_argument("--gate", type=str, default="gate", choices=["gate", "oracle"],
                    help="How a framework run routes the samples. 'gate' (default) is the inference-time gate: "
                         "different top-1 and top-2 labels go to the framework. 'oracle' is a perfect gate for "
                         "the oracle experiment: samples whose top-1 and top-2 labels are both the true label "
                         "(High retrieval) keep the original RAG, all the others (Conflicting and Low "
                         "retrieval) go to the framework. Used only with --framework.")

args = parser.parse_args()

start_batch = args.start_batch
results_path = args.results_path
knowledge_base_path = args.knowledge_base_path
test_path = args.test_path
generator_id = args.generator_id
clip_model_id = args.clip_model_id
dim_reduction = args.dim_reduction
prompt = args.prompt
top_k = args.top_k
enable_thinking = args.enable_thinking
framework = args.framework
gate = args.gate

print("Code running. CLI call:")
for _k, _v in vars(args).items():
    print(f"  {_k}: {_v}")

generator_spec = get_model_spec(generator_id)

print("Package versions:")
print(f"versions | torch {torch.__version__} | transformers {transformers.__version__} | "
      f"faiss {faiss.__version__} | sklearn {sklearn.__version__} | numpy {np.__version__}")

print("GPU device:")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"the device being used: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# if generator model does not allow multi user message, raise an error 
validate_prompt_request(generator_id, generator_spec, prompt)

# validate thinking mode request - checks whether generator model checkpoint allows. otherwise, raise an error 
validate_thinking_request(generator_id, generator_spec, enable_thinking)

# if thinking is on, this run will record "thinking" generations in results file 
thinking_on = resolve_thinking(generator_spec, enable_thinking)
print(f"the run produces thinking text: {thinking_on}")

# a framework runs only with the generator and retrieval settings it was written for - checked
# before loading any weights
if framework != "original":
    validate_framework_request(framework, generator_id, top_k, enable_thinking)
    print(f"framework: {framework} {FRAMEWORKS[framework]}")
    print(f"gate: {gate}")
# the oracle gate routes samples to a framework, so a run without one has nothing to route
elif gate == "oracle":
    raise ValueError("--gate oracle needs a --framework: with --framework original every sample runs the "
                     "original RAG and there is nothing to route.")


# read csv
knowledge_base_set = pd.read_csv(knowledge_base_path)
test_df = pd.read_csv(test_path)
# create a list of classes out of the train_set
classes_list = sorted(knowledge_base_set['true_label'].unique().tolist())

# validate image paths
validate_image_paths(knowledge_base_set["file_path"].tolist(), knowledge_base_path)
validate_image_paths(test_df["file_path"].tolist(), test_path)
# the oracle gate routes by the true label of the query
if gate == "oracle" and "true_label" not in test_df.columns:
    raise ValueError(f"--gate oracle needs a 'true_label' column in the test csv: {test_path}")

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

# initialize retrival model clip and extract embeddings to train_df
clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float16).to(device=device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id, use_fast=False)
clip_model.eval()
print(f"CLIP model dtype: {next(clip_model.parameters()).dtype}")
all_on_gpu = all(param.device.type =="cuda" for param in clip_model.parameters())
print(f"is CLIP model on GPU? {all_on_gpu}")
print(f"print model.device to check of clip has such attribute: {clip_model.device}")
print(f"print parameters device of clip: {next(clip_model.parameters()).device}")


knowledge_base_set = knowledge_base_set.reset_index(drop=True)
def get_clip_embedding(image_path, model, processor):
    """
    image_path: path of the input image.
    model: Clip Model
    processor: Clip processor
    returns: image embedding tensor
    """
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt', padding=True).to(model.device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs).pooler_output.squeeze(0)
    return image_embedding

# create embedding_list and store clip embeddings
embeddings_list= []
for kb_image_path in knowledge_base_set["file_path"]:
    embedding = get_clip_embedding(resolve_path(kb_image_path), clip_model, clip_processor)
    embedding = embedding.to(dtype=torch.float32)
    embeddings_list.append(embedding)

# use lda for dimensionality reduction
if dim_reduction == "lda":
    # stack embeddings to a matrix, (n_samples, n_features)
    X = torch.stack(embeddings_list).detach().cpu().numpy()
    # get the values of true labels
    y = knowledge_base_set["true_label"].to_numpy()
    # standardize the data before LDA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # fit LDA, None will automatically extract C-1 components
    lda = LinearDiscriminantAnalysis(n_components=None)
    kb_embeddings = lda.fit_transform(X_scaled, y)
    del embeddings_list, X, X_scaled
# if not "lda" then process clip embeddings
else:    # normalize embeddings and stor in numpy arr
    kb_embeddings = torch.stack(embeddings_list).detach().cpu().numpy()
    del embeddings_list

kb_embeddings = np.ascontiguousarray(kb_embeddings, dtype=np.float32)
faiss.normalize_L2(kb_embeddings)

# create a FAISS GPU resource manager
res = faiss.StandardGpuResources()
# use inner product on normalized embeddings to get cosine
index = faiss.GpuIndexFlatIP(res, kb_embeddings.shape[1])
# populate index with embeddings
index.add(kb_embeddings)
# create a list of indices to link to FAISS results IDs, and reset df index
id_map = list(knowledge_base_set.index)
print(f"FAISS index GPU: {type(index)}, ntotal: {index.ntotal}")
print("FAISS GPU device ID:", index.getDevice())


# load the generator
stamp("retrieval ready -> Time passed from the start of the run up to the end of retrieval")
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
print(f"[generators.registry.load_generator] generator device: {next(generator_model.parameters()).device}")
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

# what a framework needs from this run; see run_framework in frameworks/registry.py
framework_ctx = {"model": generator_model, "processor": generator_processor, "spec": generator_spec,
                 "classes_list": classes_list, "prompt": prompt, "debug": True,
                 "scoring_checked": False, "truncated_count": 0,
                 "context_exceeded_count": 0, "largest_context_tokens": 0}
# the framework columns exist only for a run that routes samples to a framework, the way the
# thinking column exists only for a run that thinks
framework_cols = (["route"] + framework_columns(framework, classes_list)) if framework != "original" else []
framework_count = 0


# create top K cols
label_cols = [f'top_label_{i+1}' for i in range(top_k)]
path_cols = [f'top_path_{i+1}' for i in range(top_k)]
cosine_cols = [f'top_cosine_{i+1}' for i in range(top_k)]
all_retrieval_cols = label_cols + path_cols + cosine_cols
print(f"label cols: {label_cols}")
print(f"all retrieval cols: {all_retrieval_cols}")

# the thinking column exists only for a run that produces reasoning; prediction is
# unaffected either way
thinking_dict = {"thinking": None} if thinking_on else {}

# if start batch not 0, then read the existing csv
if start_batch > 0:
    results_df = pd.read_csv(results_path)
# else, create predictions column and copy full test set
else:
    retrieval_dict = {col: None for col in all_retrieval_cols}
    framework_dict = {col: None for col in framework_cols}
    results_df = test_df.copy(deep=True)
    results_df = results_df.assign(prediction=None, query_file_path=None, **thinking_dict, **retrieval_dict,
                                   **framework_dict)


results_df = results_df.reset_index(drop=True)

# setting variables for inference
batch_size = 100


# if start_batch is greater than 0 then adjust df (Skip already processed rows)
if start_batch > 0:
    # calculate start row
    start_row = start_batch * batch_size
    df = test_df.iloc[start_row:].reset_index(drop=True)
else:
    df = test_df.copy(deep=True)
    df = df.reset_index(drop=True)
del test_df
# calculate how many iterations through batches are there with df length
# keep in mind the last batch might not contain "batch size" samples
num_batches = (len(df) + batch_size - 1) // batch_size

print_debug = True

check_output_truncation = True
truncated_count = 0
offloaded_count = 0
# looping through batch 0 to the last batch
for curr_batch in range(num_batches):
    # initialize start batch and end batch
    start_row = curr_batch * batch_size
    end_row = (curr_batch + 1) * batch_size
    batch_df = df.iloc[start_row:end_row].reset_index(drop=True)

    batch_predictions = []
    batch_thinking = []
    batch_framework_rows = []
    # create datasets based  on k
    top_labels_df = pd.DataFrame(columns=label_cols)
    top_paths_df = pd.DataFrame(columns=path_cols)
    top_similarities_df = pd.DataFrame(columns=cosine_cols)
    query_file_paths = []
    # memory usage
    torch.cuda.reset_peak_memory_stats()
    # looping through samples in the batch
    for _, row in batch_df.iterrows():
        # load query
        query_path = row['file_path']
        absolute_query_path = resolve_path(query_path)
        query_image = Image.open(absolute_query_path).convert('RGB')

        # extract embeddings from QUERY
        query_embedding = get_clip_embedding(absolute_query_path, clip_model, clip_processor)
        if dim_reduction == "lda":
            # standardize the clip embeddings
            query_embedding = scaler.transform(query_embedding.cpu().numpy().reshape(1, -1))
            # using trained lda to extract components for query image. shape (1, c-1)
            query_embedding = lda.transform(query_embedding)
        # is using CLIP embeddings
        else:
            query_embedding = query_embedding.detach().cpu().numpy().reshape(1, -1)

        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # run search to find top k
        scores, ids = index.search(query_embedding, top_k)
        # get a list of top k ids and top k cosine scores.
        faiss_ids = ids[0].tolist()
        top_similarities = scores[0].tolist()

        # get top k indices as presented in knowledge base set
        kb_row_indices = []
        for idx in faiss_ids:
            kb_row_indices.append(id_map[idx])

        # create top examples df
        top_examples = knowledge_base_set.loc[kb_row_indices]
        # extracting labels from top examples for saving and ensure I don't have none (less than 3)
        top_labels = top_examples['true_label'].tolist()
        top_paths = top_examples['file_path'].tolist()
        query_file_paths.append(query_path)
        while len(top_labels) < top_k:
            top_labels.append(None)
        while len(top_paths) < top_k:
            top_paths.append(None)
        while len(top_similarities) < top_k:
            top_similarities.append(None)

        # append the new row
        top_labels_df.loc[len(top_labels_df)] = top_labels
        top_paths_df.loc[len(top_paths_df)] = top_paths
        top_similarities_df.loc[len(top_similarities_df)] = top_similarities

        # route the sample: with the gate, disagreeing top-1 and top-2 labels go to the framework; with
        # the oracle, every sample that is not High retrieval does. The rest go to the original RAG below
        if framework != "original":
            if gate == "oracle":
                to_framework = oracle_needs_new_framework(top_labels, row["true_label"])
            else:
                to_framework = needs_new_framework(top_labels)
            if to_framework:
                framework_row = run_framework(framework, query_image, top_examples, framework_ctx)
                batch_framework_rows.append(framework_row)
                batch_predictions.append(framework_row["prediction"])
                batch_thinking.append(None)
                framework_count += 1
                del query_image, query_embedding, scores, ids, faiss_ids, top_similarities, kb_row_indices, top_examples
                del top_labels, top_paths, framework_row
                continue
            batch_framework_rows.append({"route": "original"})

        # build the original RAG prompt
        conversation, images = build_rag_conversation(prompt, classes_list, top_examples, query_image)

        # process inputs
        if print_debug == True:
            print("conversation:")
            print_conversation(conversation)
            print(f"conversation structure:\n{conversation}")

        # generate prediction
        prediction, thinking, gen_stats = generate_prediction(generator_model, generator_processor, conversation,
                                                              images, generator_spec,
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

        # validate promt and generation do not exceed model context window 
        context_window = get_context_window(generator_model)
        if context_window is None:
            print(f"[WARNING] pipelines.rag: '{generator_id}' does not declare a context window in "
                f"its config, so the context window validation cannot be performed for this run.")
        elif gen_stats['prompt_token_len'] + gen_stats['max_new_tokens']> context_window:
            print(f"[WARNING] pipelines.rag: the input with the assigned max_new_tokens exceed the "
            f"model context window: with {gen_stats['prompt_token_len']} input tokens "
            f"and {gen_stats['max_new_tokens']} max_new_tokens limit, against a context window of "
            f"{context_window} for model checkpoint: '{generator_id}'. Predictions may be unreliable. ")

        # generation that did not fit in device memory and only finished because its KV
        # cache was moved to host RAM. The prediction is unaffected, the sample was slower.
        if gen_stats["cache_mode"] != "gpu":
            offloaded_count += 1
            if offloaded_count == 1:
                print(f"[WARNING] pipelines.rag: the generation ran out of GPU memory and was "
                      f"retried with the KV cache offloaded to host RAM. The prediction is "
                      f"unchanged but the sample was much slower. query: {query_path}")

        # generation that ran out of budget (max new tokens) rather than finishing its answer.
        if gen_stats["finish_reason"] == "length":
            truncated_count += 1
            if truncated_count == 1:
                print(f"[WARNING] pipelines.rag: the output was truncated by max_new_tokens="
                      f"{gen_stats['max_new_tokens']}, so the answer may be incomplete. Consider "
                      f"raising max_new_tokens for '{generator_id}'."
                      f"prediction: {prediction!r}, query: {query_path}")
        elif gen_stats["finish_reason"] == "unknown":
            print(f"[WARNING] pipelines.rag: '{generator_id}' does not declare an end of "
                  f"generation token, so the truncation validation cannot be performed for this "
                  f"run.")
            # the checkpoint cannot start declaring one later, so there is nothing more to learn
            check_output_truncation = False

        batch_predictions.append(prediction)
        batch_thinking.append(thinking)
        del query_image, images, query_embedding, scores, ids, faiss_ids, top_similarities, kb_row_indices, top_examples
        del top_labels, top_paths

    # at the end of batch I will save predictions of the batch

    # validate predictions are not empty or none
    empty_count = sum(1 for p in batch_predictions if not p)
    if empty_count:
        print(f"[WARNING] {empty_count} empty predictions in this batch - generation ended "
              f"before an answer. Consider raising max_new_tokens for '{generator_id}'.")

    # update the predictions column (if there is no such column - creates it)
    # current start row (taking into account past savings)
    curr_start_row = (start_batch + curr_batch) * batch_size
    # end row (+ 100)
    curr_end_row = curr_start_row + len(batch_predictions) - 1
    # assigns the values in batch predictions to the specified rows in results df
    results_df.loc[curr_start_row:curr_end_row, "prediction"] = batch_predictions
    results_df.loc[curr_start_row:curr_end_row, "query_file_path"] = query_file_paths
    # the reasoning that produced those predictions, on the same rows
    if thinking_on:
        results_df.loc[curr_start_row:curr_end_row, "thinking"] = batch_thinking
    # saving top examples to results
    results_df.loc[curr_start_row:curr_end_row, label_cols] = top_labels_df.values
    results_df.loc[curr_start_row:curr_end_row, path_cols] = top_paths_df.values
    results_df.loc[curr_start_row:curr_end_row, cosine_cols] = top_similarities_df.values
    # saving framework results; a row that did not pass the gate does not store additional columns 
    if framework != "original":
        framework_batch_df = pd.DataFrame(batch_framework_rows, columns=framework_cols)
        results_df.loc[curr_start_row:curr_end_row, framework_cols] = framework_batch_df.values
        del framework_batch_df
    del top_labels_df, top_paths_df, top_similarities_df, batch_df, batch_thinking
    # save results
    results_df.to_csv(results_path, index=False)
    print(f"peak GPU memory this batch: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    torch.cuda.empty_cache()
    # end of batch 
    what_batch = curr_batch +1
    stamp(f"batch {what_batch}/{num_batches} done, ({len(batch_predictions)} samples) -> Time for "
          f"processing the batch")

# a few truncated samples are noise, a large share means the generation budget is too small
# for this configuration. The rate is what tells the two apart.
if truncated_count:
    print(f"[WARNING] pipelines.rag: {truncated_count} of {len(df)} samples were truncated "
          f"before the model finished its answer. Consider raising max_new_tokens for "
          f"'{generator_id}'.")

# the offloaded samples are the ones that would have ended the run before this retry
# existed. Their predictions are comparable to the rest, their timings are not.
if offloaded_count:
    print(f"[WARNING] pipelines.rag: {offloaded_count} of {len(df)} samples did not fit in "
          f"GPU memory and were generated with the KV cache offloaded to host RAM. Their "
          f"predictions are unaffected, their runtimes are not comparable to the rest.")

if framework != "original":
    routed_by = ("were not High retrieval (oracle gate)" if gate == "oracle"
                 else "had different top-1 and top-2 labels")
    print(f"routing: {framework_count} of {len(df)} samples {routed_by} and ran "
          f"framework '{framework}'; the other {len(df) - framework_count} ran the original RAG.")
    if framework_ctx["truncated_count"]:
        print(f"[WARNING] pipelines.rag: {framework_ctx['truncated_count']} framework reasoning or explanation "
              f"generations were cut off by max_new_tokens before they finished.")
    if framework_ctx["context_exceeded_count"]:
        print(f"[WARNING] pipelines.rag: {framework_ctx['context_exceeded_count']} framework generation calls "
              f"used more tokens (prompt + generated) than the context window of "
              f"{get_context_window(generator_model)} tokens for '{generator_id}' (largest: "
              f"{framework_ctx['largest_context_tokens']} tokens). Their branches are not reliable; they have "
              f"exceeds_context=True and their context size in the <branch>__context_tokens column.")

print(f"[TIME] {now_str()} | pipeline ended -> total runtime "
      f"{(time.perf_counter()-_T0)/60:.1f} min", flush=True)
