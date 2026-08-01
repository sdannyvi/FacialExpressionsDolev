import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaOnevisionForConditionalGeneration, \
    AutoProcessor
from PIL import Image
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description="Run Retrieval-Augmented Generation.")
parser.add_argument("--test_path", type=str, required=True,
                    help="the path to the text csv (the csv that needs to be classified).")
parser.add_argument("--results_path", type=str, required=True,
                    help="path to save results csv (including the name of the csv file).")
parser.add_argument("--llava_model_id", type=str, default="llava-hf/llava-v1.6-34b-hf",
                    choices=["llava-hf/llava-v1.6-34b-hf",
                             "llava-hf/llava-onevision-qwen2-7b-ov-hf",
                             "llava-hf/llava-v1.6-mistral-7b-hf"],
                    help="The path to the Hugging Face LLaVA model checkpoint.")

args = parser.parse_args()
test_path = args.test_path
results_path = args.results_path
llava_model_id = args.llava_model_id

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

# read csv
test_df = pd.read_csv(test_path).reset_index(drop=True)
# classes list
classes_list = sorted(test_df['true_label'].unique().tolist())

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
    start_row = 0

count = 0
batch_size = 100
for batch_start in range(start_row, len(test_df), batch_size):
    batch_end = min(batch_start + batch_size, len(test_df))
    batch_df = test_df.iloc[batch_start:batch_end].copy()
    batch_predictions = []
    query_file_paths = []
    # loop through images to classify
    for _, row in batch_df.iterrows():
        # load query
        query_path = row["file_path"]
        query_file_paths.append(query_path)
        with Image.open(query_path) as im:
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
        if count == 0:
            print("conversation:")
            print_conversation(conversation)
            count = count + 1
        print(f"conversation structure:\n{conversation}")
        # processing text prompt
        text_prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
        # process input
        inputs = llava_processor(images=query_image, text=text_prompt, padding=True, return_tensors="pt")

        device_type = next(iter(llava_model.parameters())).device
        inputs = inputs.to(device_type)
        print(f"device_type: {device_type}")
        # generate prediction
        with torch.no_grad():
            output = llava_model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=1.0, num_beams=1)

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = output[0][prompt_len:]
        print("prompt_len:", prompt_len)
        print("total_output_len:", output[0].shape[-1])
        print("generated_len:", gen_ids.shape[-1])

        prediction = llava_processor.decode(gen_ids, skip_special_tokens=True).strip().lower()
        print(f'prediction: {prediction}')
        batch_predictions.append(prediction)

        del inputs, output, query_image
        torch.cuda.empty_cache()
    # end batch

    # update results
    results_df.loc[batch_start:batch_end - 1, "prediction"] = batch_predictions
    results_df.loc[batch_start:batch_end - 1, "query_file_path"] = query_file_paths
    results_df.to_csv(results_path, index=False)
