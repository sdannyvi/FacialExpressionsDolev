import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/rafd_cropped.csv")

# validate cropping
print(f"number of samples in dataset: {len(df)}")
print(f"number of unique file paths is the same as length? {df['file_path'].nunique() == len(df)}")
print(f"is there a null file path? {df['file_path'].isnull().any()}")
print(f"all file paths valid?")
print((df["file_path"].str.startswith("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/RAFD/rafd_images_cropped")).all())


# unique identities
print(f"number of unique identities: {df['identity'].nunique()}")
print(f"number of samples in each identity: {df['identity'].value_counts()}")
print(f"are all identities have the same number of samples? {df['identity'].value_counts().nunique() == 1}")

unique_ids = df["identity"].unique()


# split identities (20% test, 80% train)
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42, shuffle=True)


# 3. Build train and test dataframes based on identity
train_df = df[df["identity"].isin(train_ids)]
test_df  = df[df["identity"].isin(test_ids)]

# 4. Save to CSV
train_df.to_csv("train_set_radboud.csv", index=False)
test_df.to_csv("test_set_radboud.csv", index=False)


print("are they disjoint (do not overlap in samples)")
print(set(train_df["file_path"]).isdisjoint(set(test_df["file_path"])))

# check for null identities
print("are there nulls in identity?")
print(train_df["identity"].isnull().any())
print(test_df["identity"].isnull().any())

# print unique list of identities
print(sorted(train_df["identity"].unique().tolist()))
print(sorted(test_df["identity"].unique().tolist()))

# check if train and test identities are overlap of not
print(f"are identities disjoint? {set(train_df['identity']).isdisjoint(set(test_df['identity']))}")
