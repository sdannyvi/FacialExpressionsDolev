import os.path
import sys
import pandas as pd
import numpy as np
import argparse
from Thesis.VLMs.maps import fer_plus_code
from itertools import product


# read data files
# fer_data = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer2013/icml_data.csv")

def create_datasets(input_csv_path, save_dir, set_type, threshold, match_size_with):
    """
    params:
    input_csv_path: String. path to the csv contains the full data to process
    save_dir: String. path to the directory in which the new data will be saved
    set_type: String. may receive one of two options: "kb", "public_test", "private_test"
    threshold: integer. may receive one of 3 options: 0, 50, 80
    match_size_with:

    returns:
    new dataset, with columns:
    file_path: a string path to the csv file.
    true_label: string class.
    coded_true_label: integer.
    votes_percentage: float. percentage of max vote. calculated out of filtered sum of votes (after outlier removal)
                            and note by sum = 10.
    """

    # load input csv
    input_df = pd.read_csv(input_csv_path)

    # the new dataset name
    if match_size_with is not None:
        # extract the match csv file name
        match_file_name = os.path.basename(match_size_with)
        # delete the extension of this file, keeping the name only
        match_file_name = os.path.splitext(match_file_name)[0]

        data_name = f"{set_type}_{threshold}%_[match_size_{match_file_name}].csv"
    else:
        data_name = f"{set_type}_{threshold}%.csv"

    # filter images based on set type - optional set types
    usage_map = {
        "kb": "Training",
        "public_test" : "PublicTest",
        "private_test": "PrivateTest"
    }
    # get samples of the same set type
    if set_type not in usage_map:
        raise ValueError(f"Unknown set_type: {set_type}")
    df = input_df[input_df['usage'] == usage_map[set_type]]

    # threshold filter
    rows_list = []
    classes_list = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt', 'unknown', 'NF']
    for _, row in df.iterrows():
        # keep file_path to store later
        file_path = row['file_path']

        # create emotion classes row
        row_order = df.columns[df.columns.isin(classes_list)]
        emotion_votes = row[row_order].astype(float).tolist()

        # create unknown list and emotion list for thresholding
        size = len(emotion_votes)
        emotion = [0.0] * size

        # remove all votes of 1, replace with 0
        for i in  range(size):
            if emotion_votes[i] < 1.0 + sys.float_info.epsilon:
                emotion_votes[i] = 0.0

        # calculate sum of votes
        sum_votes = sum(emotion_votes)
        # if sum votes is 0 after removal of outliers, discard image
        if sum_votes < 0.0 + sys.float_info.epsilon:
            continue

        # maximum vote
        max_vote = max(emotion_votes)

        #  filter by threshold
        if threshold == 0:
            # check if there is a tie. keep non-tie rows.
            tie = emotion_votes.count(max_vote) > 1
            if tie is False:
                emotion[np.argmax(emotion_votes)] = 1
            else:
                continue

        if threshold == 50:
            if max_vote > 0.5 * sum_votes:
                emotion[np.argmax(emotion_votes)] = 1
            # if not above the threshold, discard image
            else:
                continue

        if threshold == 80:
            if max_vote >= 0.8 * sum_votes:
                emotion[np.argmax(emotion_votes)] = 1
                # if not above the threshold, discard image
            else:
                continue

        # reach here if passes threshold
        label_idx = np.argmax(emotion)
        true_label = row_order[label_idx]
        # discard images with unknown or NF label
        if true_label in ["unknown", "NF"]:
            continue

        # get coded label and vote percentage
        coded_true_label = fer_plus_code[true_label]
        votes_percentage = round(max_vote / sum_votes, 3)

        # add row to output dataframe
        rows_list.append({
            "file_path": file_path,
            "true_label": true_label,
            "coded_true_label": coded_true_label,
            "votes_percentage": votes_percentage
        })
        ## end loop

    # convert into dataframe
    output_df = pd.DataFrame(rows_list)

    # if equal size needed
    if match_size_with is not None:
        # get size of sample
        match_df = pd.read_csv(match_size_with)
        match_size = len(match_df)
        current_size = len(output_df)
        # if current dataset is larger, randomly select rows in match_size
        if current_size > match_size:
            output_df = output_df.sample(n=match_size, random_state=42).reset_index(drop=True)
        else:
            print(f"Current dataset has {current_size} samples, which is less or equal to match size ({match_size})."
                  f"No sampling needed.")

    # save to csv
    path = os.path.join(save_dir,data_name)
    output_df.to_csv(path, index=False)
    print(f"file name: {data_name}, was successfully saved as a csv file in path: {path}")


# create sets
list_types = ["kb", "public_test", "private_test"]
list_thresh = [0,50,80]
input_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/fer_plus_data.csv"
save_path = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets"
for set_t, thresh,in product(list_types, list_thresh):
    create_datasets(input_csv_path=input_path,
                    save_dir=save_path,
                    set_type=set_t,
                    threshold=thresh,
                    match_size_with=None
                    )


# KB threshold 50 (equal size to: KB threshold 80)
create_datasets(input_csv_path=input_path,
                    save_dir=save_path,
                    set_type="kb",
                    threshold=50,
                    match_size_with="/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/kb_80%.csv"
                    )