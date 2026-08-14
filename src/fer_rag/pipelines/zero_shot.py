import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from PIL import Image
import pandas as pd
import argparse
import sys

from ..generators import AVAILABLE_MODELS, load_generator, generate_prediction

parser = argparse.ArgumentParser(description="Run Retrieval-Augmented Generation.")
parser.add_argument("--test_path", type=str, required=True,
                    help="the path to the text csv (the csv that needs to be classified).")
parser.add_argument("--results_path", type=str, required=True,
                    help="path to save results csv (including the name of the csv file).")
parser.add_argument("--llava_model_id", type=str, default="llava-hf/llava-v1.6-34b-hf",
                    choices=AVAILABLE_MODELS,
                    help="The path to the Hugging Face generator model checkpoint.")

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

# load the generator
llava_model, llava_processor, generator_spec = load_generator(llava_model_id)

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

debug_printed = False
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
        if debug_printed == False:
            print("conversation:")
            print_conversation(conversation)
            print(f"conversation structure:\n{conversation}")
 
        # generate prediction
        prediction, gen_stats = generate_prediction(llava_model, llava_processor, conversation, [query_image], generator_spec)


        if debug_printed == False: 
            print(f"gen_stats: {gen_stats}")
            print(f'prediction: {prediction}')
            debug_printed = True

        batch_predictions.append(prediction)

        del query_image
        torch.cuda.empty_cache()
    # end batch

    # update results
    results_df.loc[batch_start:batch_end - 1, "prediction"] = batch_predictions
    results_df.loc[batch_start:batch_end - 1, "query_file_path"] = query_file_paths
    results_df.to_csv(results_path, index=False)
