import os
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import pandas as pd
from accelerate import Accelerator
import argparse


def llava_zero_shot(start_batch=0, results_path=None,test_path=None, model_hub="llava-hf/llava-v1.6-mistral-7b-hf"):
    """
    all paths must be absolute paths.
    params:
    start batch (int): the batch number to start from. if batch 0 then starting from the first row in test_set (set to classify).
                    if batch is 2 then starting from the row 200 of test_set.
    results_path (str): the path in which the csv results will be saved. given the full path and the csv name.
                        if start batch > 0, results_path must be the path to the existing results csv.
    test_path (str): the path to the test set, set I want to classify
    """

    # set accelerator for multi-GPU
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    accelerator = Accelerator(device_placement=True)
    # check if CUDA is being used
    if accelerator.device.type == "cuda":
        print("CUDA is being used.")


    # initialize llava model and processor
    llava_model = LlavaNextForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_hub,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    llava_model = accelerator.prepare(llava_model)
    llava_model.eval()
    llava_processor = LlavaNextProcessor.from_pretrained(
        pretrained_model_name_or_path=model_hub,
        use_fast=True
    )
    llava_model.eval()

    # read csv
    test_df = pd.read_csv(test_path)
    # classes list
    classes_list = test_df['true_label'].unique().tolist()

    # if start batch not 0, then read the existing csv
    if start_batch > 0:
        results_df = pd.read_csv(results_path)
    # else, create predictions column and copy full test set
    else:
        results_df = test_df.copy(deep=True)
        results_df["predictions"] = None

    results_df = results_df.reset_index(drop=True)

    # initialize batch size and check if start batch greater than 0, then adjust the first row of test df
    batch_size = 100
    if start_batch > 0:
        # calculate start row
        start_row = start_batch * batch_size
        df = test_df.iloc[start_row:].reset_index(drop=True)
    else:
        df = test_df.copy(deep=True).reset_index(drop=True)


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
        # loop through images to classify
        for _, row in batch_df.iterrows():
            # loading image and convert to RGB
            image_path = row['file_path']
            query_image = Image.open(image_path).convert('RGB')

            # infer with llava
            query_text_prompt = (f"This image shows a person expressing an emotion."
                           f"Please analyze the emotion in the image and select the best match from the following options: {', '.join(classes_list)}."
                           f"Respond with only one word: the emotion name.")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query_text_prompt},
                    ]
                }
            ]

            # processing text prompt
            text_prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
            # process input
            inputs = llava_processor(images=query_image, text=text_prompt, padding=True, return_tensors="pt").to(llava_model.device)

            # generate prediction
            with torch.no_grad():
                output = llava_model.generate(**inputs, max_new_tokens=10)
            prediction = llava_processor.decode(output[0], skip_special_tokens=True).strip()
            print(f'prediction: {prediction}')

            # pre-process prediction output
            if model_hub == "llava-hf/llava-v1.6-mistral-7b-hf":
                pred_parts = prediction.split("[/INST]")
            elif model_hub == "llava-hf/llava-v1.6-34b-hf":
                pred_parts = prediction.split("\n")
            prediction = pred_parts[-1].strip().lower()
            batch_predictions.append(prediction)
            print(f"processed prediction {prediction}")


            del inputs, output, query_image
            torch.cuda.empty_cache()
        ## end of batch
        # at the end of batch I will save predictions of the batch

        # update the predictions column (if there is no such column - creates it)
        # current start row (taking into account past savings)
        curr_start_row = (start_batch + curr_batch) * batch_size
        # end row (+ 100)
        curr_end_row = curr_start_row + len(batch_predictions) - 1
        # assigns the values in batch predictions to the specified rows in results df
        results_df.loc[curr_start_row:curr_end_row, "predictions"] = batch_predictions
        # save results
        results_df.to_csv(results_path, index=False)



# # # FER2013
# # calling the function
# test_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/LLaVa/llava_rag/kb_sets/combined_test.csv"
# save_to_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/LLaVa/llava_rag/results/llava_zero_shot_34b.csv"
#
# llava_zero_shot(start_batch=0, results_path=save_to_path,test_path=test_path, model_hub="llava-hf/llava-v1.6-34b-hf")

#
# # # # FER PLUS
# test_path_fer_plus = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/rag_sets/fer_plus_test.csv"
# save_to_path_fer_plus = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/llava_results/fer_plus_llava_zero_shot_34b.csv"
# llava_zero_shot(start_batch=0, results_path=save_to_path_fer_plus,test_path=test_path_fer_plus, model_hub="llava-hf/llava-v1.6-34b-hf")