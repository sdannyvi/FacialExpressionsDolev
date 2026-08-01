import pandas as pd
import numpy as np

# read data files
icml_data = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer2013/icml_data.csv")
fer_plus_data = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/fer2013new.csv")

# change the paths of the icml data
icml_data['file_path'] = icml_data['file_path'].str.replace("Vision-Language-Models", "VLMs")
print(icml_data['file_path'].head())


# print columns for each data
print(f"icml data columns: \n {icml_data.columns.tolist()}")
print(f"fer plus columns: \n {fer_plus_data.columns.tolist()}")

# print data types of columns
print(f"icml data types: \n {icml_data.dtypes}")
print(f"fer plus types: \n {fer_plus_data.dtypes}")

# reset indices before concatenation
icml_data.reset_index(drop=True, inplace=True)
fer_plus_data.reset_index(drop=True, inplace=True)

# concatenate dataframes
combined_df = pd.concat([icml_data, fer_plus_data], axis=1)

# if all values of usage columns are the same, now check if the columns store the exact same information.
# if yes, then drop one of them
print(f"the values in Usage: {combined_df['Usage'].unique()}, "
      f"the values in usage {combined_df['usage'].unique()}")
all_equal = (combined_df['Usage'] == combined_df['usage']).sum() == len(combined_df)
print(f"are the column usage and Usage identical? {all_equal}")

# drop unnecessary columns
combined_df.drop(columns=['Usage', 'Image name', 'true_label', 'coded_true_label'], inplace=True)

print(f"combined data columns: \n{combined_df.columns.tolist()}")
print(f"combined data dtypes: \n{combined_df.dtypes}")


# nulls
count = 0
for col in combined_df.columns:
    if combined_df[col].isnull().sum() != 0:
        print(f"NULLS in column '{col}': {combined_df[col].isnull().sum()}")
    else:
        count += 1
if count == len(combined_df.columns.tolist()):
    print("No nulls in none of the columns.")

# map column names to match fer 2013 and all other runs class names (i.e. 'happiness' becomes 'happy')
print(f"columns of combined fer plus: \n{combined_df.columns.tolist}")
label_map_match_2013 = {
    'neutral': 'neutral',
    'happiness': 'happy',
    'surprise': 'surprise',
    'sadness': 'sad',
    'anger': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'unknown': 'unknown',
    'contempt': 'contempt'
}
combined_df.rename(columns={col: label_map_match_2013[col] for col in combined_df.columns if col in label_map_match_2013}, inplace=True)
print(f"columns of combined fer plus, after matching columns names to fer2013 classes: \n{combined_df.columns.tolist}")
combined_df.to_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/fer_plus_data.csv", index=False)