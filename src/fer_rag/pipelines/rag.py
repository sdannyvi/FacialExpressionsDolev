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

from ..generators import AVAILABLE_MODELS, get_model_spec, load_generator, generate_prediction


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
print(f"Code running:\nGenerator model: {generator_id},\nCLIP model: {clip_model_id},\ndim_reduction: {dim_reduction},\n"
      f"prompt: {prompt},\ntop_k: {top_k}")

# validate the requested prompt structure against the chosen generator before doing any
# work, so an unsupported combination fails immediately instead of after loading weights
generator_spec = get_model_spec(generator_id)
if prompt == "multi-user-message" and not generator_spec.get("supports_consecutive_user", True):
    raise ValueError(
        f"'{generator_id}' requires its chat roles to alternate, so it cannot be used with "
        f"--prompt multi-user-message. Use --prompt single-user-message instead."
    )

# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"the device being used: {device}")

# read csv
knowledge_base_set = pd.read_csv(knowledge_base_path)
test_df = pd.read_csv(test_path)
# create a list of classes out of the train_set
classes_list = sorted(knowledge_base_set['true_label'].unique().tolist())

# validate image paths
validate_image_paths(knowledge_base_set["file_path"].tolist(), knowledge_base_path)
validate_image_paths(test_df["file_path"].tolist(), test_path)

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
        image_embedding = model.get_image_features(**inputs).squeeze(0)
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
generator_model, generator_processor, generator_spec = load_generator(generator_id)


# create top K cols
label_cols = [f'top_label_{i+1}' for i in range(top_k)]
path_cols = [f'top_path_{i+1}' for i in range(top_k)]
cosine_cols = [f'top_cosine_{i+1}' for i in range(top_k)]
all_retrieval_cols = label_cols + path_cols + cosine_cols
print(f"label cols: {label_cols}")
print(f"all retrieval cols: {all_retrieval_cols}")

# if start batch not 0, then read the existing csv
if start_batch > 0:
    results_df = pd.read_csv(results_path)
# else, create predictions column and copy full test set
else:
    retrieval_dict = {col: None for col in all_retrieval_cols}
    results_df = test_df.copy(deep=True)
    results_df = results_df.assign(prediction=None, query_file_path=None, **retrieval_dict)


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

debug_printed = False
# looping through batch 0 to the last batch
for curr_batch in range(num_batches):
    # initialize start batch and end batch
    start_row = curr_batch * batch_size
    end_row = (curr_batch + 1) * batch_size
    batch_df = df.iloc[start_row:end_row].reset_index(drop=True)

    batch_predictions = []
    # create datasets based  on k
    top_labels_df = pd.DataFrame(columns=label_cols)
    top_paths_df = pd.DataFrame(columns=path_cols)
    top_similarities_df = pd.DataFrame(columns=cosine_cols)
    query_file_paths = []
    # looping through samples in the batch
    for _, row in batch_df.iterrows():
        # memor usage
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated() / 1024 ** 3
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

        # initialize vars for inference
        conversation = []
        images = []
        # if prompt is multiple user message
        if prompt == "multi-user-message":
            # loop through examples
            for _, example_row in top_examples.iterrows():
                example_label = example_row['true_label']
                example_image = Image.open(resolve_path(example_row['file_path'])).convert('RGB')

                example_prompt_text = (f"Example: This image shows a person expressing the emotion: "
                                       f"'{example_label}'.")
                conversation.append(
                    {"role": "user",
                     "content": [
                         {"type": "image"},
                         {"type": "text", "text": example_prompt_text}
                     ]}
                )
                images.append(example_image)

            # query image prompt
            query_prompt_text = (f"This image also shows a person expressing an emotion."
                                 f"Based on the examples provided, please analyze the emotion in this image and "
                                 f"select the best match from"
                                 f"the following options: {', '.join(classes_list)}."
                                 f"Respond with only one word: the emotion name.")

            conversation.append(
                {"role": "user",
                 "content": [
                     {"type": "image"},
                     {"type": "text", "text": query_prompt_text}
                 ]}
            )
            images.append(query_image)
        elif prompt == "single-user-message":
            # add a system message
            conversation.append({
                "role": "system",
                "content": [
                    {"type": "text",
                     "text": f"You are an expert in classifying emotions from facial expressions in images.\n"
                             f"You are given {top_k} example images with their corresponding emotion labels, followed by a query image.\n"
                             f"Based on the examples provided, analyze the facial expression in the query "
                             f"image and classify the emotion. Follow the user's requested output format."
                     }
                ]
            })

            content = []
            user_text = [f"You are given {top_k} labeled examples and one query image, in the following order:"]

            for i, example_row in enumerate(top_examples.itertuples(), start=1):
                print(f"example row: {example_row}")
                # add images to image list
                example_label = example_row.true_label
                example_image = Image.open(resolve_path(example_row.file_path)).convert('RGB')
                images.append(example_image)

                # add user text
                content.append({"type": "image"})
                user_text.append(f"Image {i} label: {example_row.true_label}")

            # add query image
            images.append(query_image)
            content.append({"type": "image"})
            user_text.append(f"Image {top_k + 1} is the query. Based on the examples, classify the emotion shown in this image into one of the following emotions: {', '.join(classes_list)}.")
            user_text.append("Respond with only one word: the emotion label.")
            print(f"user text: {user_text}")
            # concat user text
            user_text = "\n".join(user_text)
            content.append({"type": "text", "text": user_text})
            conversation.append({"role": "user", "content": content})

        # process inputs
        if debug_printed == False:
            print("conversation:")
            print_conversation(conversation)
            print(f"conversation structure:\n{conversation}")

        # generate prediction
        prediction, gen_stats = generate_prediction(generator_model, generator_processor, conversation, images, generator_spec)

        if debug_printed == False: 
            print(f"gen_stats: {gen_stats}")
            print(f'prediction: {prediction}')
            debug_printed = True

        batch_predictions.append(prediction)
        del query_image, images, query_embedding, scores, ids, faiss_ids, top_similarities, kb_row_indices, top_examples
        del top_labels, top_paths

    # at the end of batch I will save predictions of the batch

    # update the predictions column (if there is no such column - creates it)
    # current start row (taking into account past savings)
    curr_start_row = (start_batch + curr_batch) * batch_size
    # end row (+ 100)
    curr_end_row = curr_start_row + len(batch_predictions) - 1
    # assigns the values in batch predictions to the specified rows in results df
    results_df.loc[curr_start_row:curr_end_row, "prediction"] = batch_predictions
    results_df.loc[curr_start_row:curr_end_row, "query_file_path"] = query_file_paths
    # saving top examples to results
    results_df.loc[curr_start_row:curr_end_row, label_cols] = top_labels_df.values
    results_df.loc[curr_start_row:curr_end_row, path_cols] = top_paths_df.values
    results_df.loc[curr_start_row:curr_end_row, cosine_cols] = top_similarities_df.values
    del top_labels_df, top_paths_df, top_similarities_df, batch_df
    # save results
    results_df.to_csv(results_path, index=False)
    torch.cuda.empty_cache()
