import os
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
from accelerate import Accelerator
from sentence_transformers import util
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
import faiss
import numpy as np


# parameters
parser = argparse.ArgumentParser(description="Run LLaVA in RAG framework, using CLIP as a retriever.")
parser.add_argument('--start_batch', type=int, default=0,
                    help="Batch number to start from. if batch 0 then starting from the first row in test set. "
                         "if batch is 2 then  starting classify from row 200 in test set.")
parser.add_argument("--results_path", type=str, required=True,
                    help="path to save results csv (including the name of the csv file).")
parser.add_argument("--knowledge_base_path", type=str, required=True,
                    help="the path to the knowledge base csv file.")
parser.add_argument("--test_path", type=str, required=True,
                    help="the path to the text csv (the csv that needs to be classified).")
parser.add_argument("--model_hub_llava", type=str, default="llava-hf/llava-v1.6-34b-hf",
                    help="the path to hugging face llava model (choose the model size)."
                         "the smallest I used overall: 'llava-hf/llava-v1.6-mistral-7b-hf'.")
parser.add_argument("--model_hub_clip", type=str, default="openai/clip-vit-large-patch14",
                    help="the path to hugging face CLIP model (choose the model size)."
                         "the smallest I used overall: 'openai/clip-vit-base-patch32'.")
parser.add_argument("--dim_reduction",type=str,  default="lda",
                    help="whether to use LDA or none dimensionality reduction method at all. "
                         "currently excepting the following values: lda, none")


args = parser.parse_args()

start_batch = args.start_batch
results_path = args.results_path
knowledge_base_path = args.knowledge_base_path
test_path = args.test_path
model_hub_llava = args.model_hub_llava
model_hub_clip = args.model_hub_clip
dim_reduction = args.dim_reduction


# read csv
knowledge_base_set = pd.read_csv(knowledge_base_path)
test_df = pd.read_csv(test_path)


# create a list of classes out of the train_set
classes_list = sorted(knowledge_base_set['true_label'].unique().tolist())


# Set up Accelerator for multi-GPU support
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
accelerator = Accelerator(device_placement=True)
# print device
# Check if CUDA is being used
if accelerator.device.type == "cuda":
    print("CUDA is being used.")
    if torch.cuda.is_available():
        print("GPU IDs being used by Accelerator")



# initialize retrival model clip and extract embeddings to train_df
clip_model = CLIPModel.from_pretrained(model_hub_clip)
clip_model.to(dtype=torch.float16, device=accelerator.device)
clip_model = accelerator.prepare(clip_model)
clip_processor = CLIPProcessor.from_pretrained(model_hub_clip)
clip_model.eval()


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
    embedding = get_clip_embedding(kb_image_path, clip_model, clip_processor)
    embedding = embedding.to(dtype=torch.float32, device='cuda')
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
    X_lda = lda.fit_transform(X_scaled, y)

    # normalize for cosine
    X_lda = X_lda.astype("float32")
    print("X_lda type:", type(X_lda))
    print("X_lda dtype:", getattr(X_lda, "dtype", None))
    print("X_lda shape:", getattr(X_lda, "shape", None))
    print("X_lda is contiguous:", getattr(X_lda, "flags", {}).c_contiguous if hasattr(X_lda, "flags") else None)
    faiss.normalize_L2(X_lda)
    kb_embeddings = X_lda.copy()
# if not "lda" then process clip embeddings
else:
    # normalize embeddings and stor in numpy arr
    embeddings_matrix = torch.stack(embeddings_list)
    embeddings_matrix = torch.nn.functional.normalize(embeddings_matrix, p=2, dim=1)
    kb_embeddings = embeddings_matrix.detach().cpu().numpy().astype("float32")

kb_embeddings = np.ascontiguousarray(kb_embeddings, dtype=np.float32)
# create a FAISS GPU resource manager
res = faiss.StandardGpuResources()
# use inner product on normalized embeddings to  get cosine
index = faiss.GpuIndexFlatIP(res, kb_embeddings.shape[1])
# populate index with embeddings
index.add(kb_embeddings)
# create a list of indices to link to FAISS results IDs, and reset df index
id_map = list(knowledge_base_set.index)
# delete embedding embedding list
del embeddings_list,X, X_scaled, X_lda, kb_embeddings


# initializing the llava model and locating it in GPUs
llava_model = LlavaNextForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_hub_llava,
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"device map: {llava_model.hf_device_map}")
llava_model = accelerator.prepare(llava_model)
llava_processor = LlavaNextProcessor.from_pretrained(
    pretrained_model_name_or_path=model_hub_llava,
    use_fast=True
)
llava_model.eval()



# if start batch not 0, then read the existing csv
if start_batch > 0:
    results_df = pd.read_csv(results_path)
# else, create predictions column and copy full test set
else:
    results_df = test_df.copy(deep=True)
    results_df = results_df.assign(prediction=None, top_label_1=None, top_label_2=None, top_label_3=None,
                                   top_path_1=None, top_path_2=None, top_path_3=None,
                                   top_cosine_1=None, top_cosine_2=None,
                                   top_cosine_3=None)


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


# looping through batch 0 to the last batch
for curr_batch in range(num_batches):
    # initialize start batch and end batch
    start_row = curr_batch * batch_size
    end_row = (curr_batch + 1) * batch_size
    batch_df = df.iloc[start_row:end_row].reset_index(drop=True)

    batch_predictions = []
    top_labels_df = pd.DataFrame(columns=['top_label_1', 'top_label_2', 'top_label_3'])
    top_paths_df = pd.DataFrame(columns=['top_path_1', 'top_path_2','top_path_3'])
    top_similarities_df = pd.DataFrame(columns=['top_cosine_1','top_cosine_2','top_cosine_3'])
    query_file_paths = []
    # looping through samples in the batch
    for _, row in batch_df.iterrows():
        query_path = row['file_path']
        query_image = Image.open(query_path).convert('RGB')

        # extract embeddings from QUERY
        query_embedding = get_clip_embedding(query_path, clip_model, clip_processor)
        if dim_reduction == "lda":
            # standardize the clip embeddings
            query_embedding = scaler.transform(query_embedding.cpu().numpy().reshape(1, -1))
            # using trained lda to extract components for query image. shape (1, c-1)
            query_embedding = lda.transform(query_embedding).astype("float32")
            # normalize embedding, shape (1, c-1) numpy array
            faiss.normalize_L2(query_embedding)
        # is using CLIP embeddings
        else:
            query_embedding = query_embedding.to(dtype=torch.float32)
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
            # shape (1,d) np array
            query_embedding = query_embedding.detach().cpu().numpy().astype("float32").reshape(1, -1)

        # run search to find top k
        k=3
        scores, ids = index.search(query_embedding, k)
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
        while len(top_labels) < k:
            top_labels.append(None)
        while len(top_paths) < k:
            top_paths.append(None)
        while len(top_similarities) <k:
            top_similarities.append(None)

        # append the new row
        top_labels_df.loc[len(top_labels_df)] = top_labels
        top_paths_df.loc[len(top_paths_df)] = top_paths
        top_similarities_df.loc[len(top_similarities_df)] = top_similarities
        # initialize vars for inference
        conversation = []
        images = []

        # loop through examples
        for _, example_row in top_examples.iterrows():
            example_label = example_row['true_label']
            example_image = Image.open(example_row['file_path']).convert('RGB')

            example_prompt_text = (f"Example: This image shows a person expressing the emotion: '{example_label}'.")
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
                             f"Based on the examples provided, please analyze the emotion in this image and select the best match from"
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

        text_prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = llava_processor(images=images, text=text_prompt, padding=True, return_tensors="pt").to(llava_model.device)

        # generate prediction
        with torch.no_grad():
            output = llava_model.generate(**inputs, max_new_tokens=10)
        prediction = llava_processor.decode(output[0], skip_special_tokens=True).strip()
        print(f'prediction: {prediction}')


        # pre-process prediction output
        if model_hub_llava == "llava-hf/llava-v1.6-mistral-7b-hf":
            pred_parts = prediction.split("[/INST]")
        elif model_hub_llava == "llava-hf/llava-v1.6-34b-hf":
            pred_parts = prediction.split("\n")
        prediction = pred_parts[-1].strip().lower()
        batch_predictions.append(prediction)
        print(f"processed prediction {prediction}")

        del inputs, output, query_image, images, query_embedding, scores, ids, faiss_ids, top_similarities, kb_row_indices, top_examples
        del top_labels, top_paths
        torch.cuda.empty_cache()
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
    results_df.loc[curr_start_row:curr_end_row,["top_label_1", "top_label_2", "top_label_3"]] = top_labels_df.values
    results_df.loc[curr_start_row:curr_end_row,["top_path_1", "top_path_2","top_path_3"]] = top_paths_df.values
    results_df.loc[curr_start_row:curr_end_row,["top_cosine_1","top_cosine_2","top_cosine_3"]] = top_similarities_df.values
    del top_labels_df, top_paths_df, top_similarities_df, batch_df
    # save results
    results_df.to_csv(results_path, index=False)



