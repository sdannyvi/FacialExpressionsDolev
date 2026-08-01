"""
each time takes different proportion out of the train set, where each class samples proportionally,
and each sample not sampled more than once, the knowledge base size grows without
resampling from previous knowledge base versions
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os


def create_knowledge_base(sizes, data_name, full_df, sampling_method, save_dir=""):
    """
    sizes (list): list of sizes to take out of the dataset (train set path)
    data_name (String): the name of the dataset as string to save as csv.  example: fer_plus
    full_df (DataFrame): a dataframe of the full dataset.
    stratified (boolean): if I want to sample stratified then True, for equal sampling - set to False.
    save_dir (string): path to the directory where the knowledge base dataset will be saved as csv
    sampling_method: "stratify", "equal", "random".
    """

    # split dataset to train and test // to make sure samples are left in all classes
    df, reserved_df = train_test_split(full_df, test_size=0.2,
                                         stratify=full_df["true_label"],
                                         random_state=42)

    # prev knowledge base (from prv steps)
    previous_kb  = pd.DataFrame(columns=full_df.columns)

    for size in sizes:
        # create a path to save knowledge base
        save_path_kb = os.path.join(save_dir, f"{data_name}_kb_{size}.csv")

        # number of samples to sample
        percentage_to_sample = size - (len(previous_kb) / len(full_df))
        num_samples = int(percentage_to_sample * len(full_df))

        # filter out samples that were already sampled before
        available_df = df[~df["file_path"].isin(previous_kb["file_path"])]

        # if stratify
        if sampling_method=='stratify':
            # percentage of each class
            label_distribution = available_df["true_label"].value_counts(normalize=True)

            # number of samples to  sample from each class
            samples_per_class = (label_distribution * num_samples).round().astype(int)


            # loop through labels and samples the amount needed proportionally
            for label, n_samples in samples_per_class.items():
                # take the relevant label
                group = available_df[available_df["true_label"] == label]
                # sample from the group where the minimum is the amount needed or available
                sampled = group.sample(n=min(n_samples, len(group)), random_state=42)
                if previous_kb.empty:
                    previous_kb = sampled.copy(deep=True).reset_index(drop=True)
                else:
                    previous_kb = pd.concat([previous_kb,sampled])
        # if not, sample equally from all classes
        elif sampling_method=='equal':
            # use num_samples to calculate how many samples to sample from each class
            num_classes = full_df["true_label"].nunique()
            samples_per_class = num_samples // num_classes

            for label, group in available_df.groupby("true_label"):
                sampled = group.sample(n=min(samples_per_class, len(group)), random_state=42)
                if previous_kb.empty:
                    previous_kb = sampled.copy(deep=True).reset_index(drop=True)
                else:
                    previous_kb = pd.concat([previous_kb,sampled])
        elif sampling_method=='random':
            sampled = available_df.sample(n=num_samples, random_state=42)
            if previous_kb.empty:
                previous_kb = sampled.copy(deep=True).reset_index(drop=True)
            else:
                previous_kb = pd.concat([previous_kb, sampled])



        # shuffle
        previous_kb = previous_kb.sample(frac=1, random_state=42).reset_index(drop=True)


        # save
        previous_kb.to_csv(save_path_kb, index=False)
        print(f"knowledge Base with size {size} saved to {save_path_kb}")
        print(f"number of samples in each class: {previous_kb['true_label'].value_counts()}")
        print(f"Knowledge base with size: {size} percentage of samples"
              f" in kb saved out of the total df is: {len(previous_kb) / len(full_df)}")
        print(f"knowledge base size:{len(previous_kb)},"
              f"compared to number of unique rows: {previous_kb['file_path'].nunique()}")

    # //end of outer loop sizes
    # concat available df and reserved df as test
    available_df =  df[~df["file_path"].isin(previous_kb["file_path"])].copy(deep=True).reset_index(drop=True)
    test_df = pd.concat([available_df,reserved_df])
    # save test df
    save_path_test = os.path.join(save_dir, f"{data_name}_test.csv")
    test_df.to_csv(save_path_test, index=False)
    print(f"test set was saved to path: {save_path_test}")
    print(f"number of samples in each class in test set: {test_df['true_label'].value_counts()}")
    print(f"the relative size of knowledge bases and test is: \n")
    for size in sizes:
        save_path_kb = os.path.join(save_dir, f"{data_name}_kb_{size}.csv")
        kb_df = pd.read_csv(save_path_kb)
        print(f"{size}: kb is {len(kb_df) / len(test_df)} out of test set.")

    return test_df


# fer_plus_data = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/fer_clean_confidence_0.6.csv")
# save_to = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/Vision-Language-Models/data/fer+/rag_sets"
# create_knowledge_base(sizes=[0.05,0.1], data_name="fer_plus", full_df=fer_plus_data,
#                       stratified=True, save_dir=save_to)




