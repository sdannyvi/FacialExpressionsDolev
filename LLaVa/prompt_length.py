import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import (LlavaNextProcessor, LlavaNextForConditionalGeneration, CLIPProcessor, CLIPModel,
                          LlavaOnevisionForConditionalGeneration, AutoProcessor)
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
parser = argparse.ArgumentParser(description="Run Retrieval-Augmented Generation.")
parser.add_argument("--knowledge_base_path", type=str, required=True,
                    help="the path to the knowledge base csv file.")
parser.add_argument("--llava_model_id", type=str, default="llava-hf/llava-v1.6-34b-hf",
                    help="the path to hugging face llava model (choose the model size)."
                         "the smallest I used overall: 'llava-hf/llava-v1.6-mistral-7b-hf'.")
parser.add_argument("--prompt",type=str,  default="single-user-message",
                    choices=["single-user-message", "multi-user-message"],
                    help="options: single-user-message, multi-user-message")
parser.add_argument("--top_k",type=int, default=2,
                    help="Number of top examples to retrieve from the knowledge base.")

args = parser.parse_args()
knowledge_base_path = args.knowledge_base_path
llava_model_id = args.llava_model_id
prompt = args.prompt
top_k = args.top_k
print(f"LLaVA Model: {llava_model_id}")
# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"the device being used: {device}")


# load llava
if "next" in llava_model_id.lower() or "v1.6" in llava_model_id.lower():
    llava_model = LlavaNextForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=llava_model_id,
        torch_dtype=torch.float16,
        device_map="balanced"
    )
    llava_processor = LlavaNextProcessor.from_pretrained(
        pretrained_model_name_or_path=llava_model_id,
        use_fast=True)
elif "onevision" in llava_model_id.lower():
    llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=llava_model_id,
        torch_dtype=torch.float16,
        device_map="balanced",
    )
    llava_processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=llava_model_id,
        use_fast=True
    )
else:
    raise ValueError(f"Invalid model id '{llava_model_id}': not a supported model.")


llava_model.eval()
print(f"llava model dtype: {next(llava_model.parameters()).dtype}")
print(f"llava device map: {llava_model.hf_device_map}")


# print the max size length
max_context_length = llava_model.config.text_config.max_position_embeddings
print(f"model max length: {max_context_length}")
print(f"Model Memory Footprint: {llava_model.get_memory_footprint() / 1024**3:.2f} GB")
#### configuration ####
print("temperature =", llava_model.generation_config.temperature)
print("do sample =", llava_model.generation_config.do_sample)
print("num beams =", llava_model.generation_config.num_beams)
print(f"padding side: {llava_processor.tokenizer.padding_side}")
print(f"Default return_dict_in_generate: {llava_model.generation_config.return_dict_in_generate}")
print(f"Default output_scores: {llava_model.generation_config.output_scores}")


# read csv
knowledge_base_set = pd.read_csv(knowledge_base_path)
# create a list of classes out of the train_set
classes_list = sorted(knowledge_base_set['true_label'].unique().tolist())


# loading query
query_path = knowledge_base_set.iloc[0]["file_path"]
query_label = knowledge_base_set.iloc[0]["true_label"]
query_image = Image.open(query_path).convert('RGB')

# loading examples
indices = []
for idx in range(0+10, top_k+10):
    indices.append(idx)
top_examples = knowledge_base_set.iloc[indices]

# get file path and labels of top examples
top_labels = top_examples['true_label'].tolist()
top_paths = top_examples['file_path'].tolist()

# prompt
conversation = []
images = []

# if prompt is multiple user message
if prompt == "multi-user-message":
    # loop through examples
    for _, example_row in top_examples.iterrows():
        example_label = example_row['true_label']
        example_image = Image.open(example_row['file_path']).convert('RGB')

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
        example_image = Image.open(example_row.file_path).convert('RGB')
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

print(f"conversation structure: {conversation}")
text_prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = llava_processor(images=images, text=text_prompt, padding=True, return_tensors="pt")
device_type = next(iter(llava_model.parameters())).device
inputs = inputs.to(device_type)
print(f"device_type: {device_type}")
print(f"llava device map: {llava_model.hf_device_map}")

# ==========================================
#      CHECK TOKEN LENGTH (Total)
# ==========================================
# inputs['input_ids'] shape is [batch_size, sequence_length]
# Since batch_size is 1 here, the sequence_length is the total token count.
total_input_tokens = inputs['input_ids'].shape[1]

print(f" TOKEN LIMIT CHECK")
print("="*40)
print(f"Total Input Tokens (Text + Image embeddings): {total_input_tokens}")
print(f"Max tokens allowed for this model: {llava_processor.tokenizer.model_max_length}")

if total_input_tokens > max_context_length:
    print(f"\nCRITICAL WARNING: Input exceeds limit by {total_input_tokens - max_context_length} tokens!")
    print("Generation may fail or truncate inputs.")
else:
    print(f"\nStatus: SAFE. You have {max_context_length - total_input_tokens} tokens remaining.")

# Decode back the text portion after tokenization
decoded_text = llava_processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
print("===== text prompt =====")
print(decoded_text)
#################################################



# generate prediction
with torch.no_grad():
    output = llava_model.generate(**inputs, max_new_tokens=10, do_sample=False, temperature=1.0, num_beams=1,
                                  output_scores=False, return_dict_in_generate=False)
generated_ids = output[:, inputs["input_ids"].shape[1]:]
prediction_clean = llava_processor.decode(generated_ids[0], skip_special_tokens=True).strip().lower()
prediction = llava_processor.decode(output[0], skip_special_tokens=True).strip()
print(f'raw prediction: {prediction}')
print(f"processed prediction 1: {prediction_clean}")
# pre-process prediction output
if llava_model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
    pred_parts = prediction.split("[/INST]")
elif llava_model_id == "llava-hf/llava-v1.6-34b-hf":
    pred_parts = prediction.split("\n")
elif llava_model_id == "llava-hf/llava-onevision-qwen2-7b-ov-hf":
    pred_parts = prediction.split("assistant")
prediction = pred_parts[-1].strip().lower()
print(f"processed prediction 2: {prediction}")