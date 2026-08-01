"""
storing constant values across all experiments - label strings, code etc.
"""
import pandas as pd
import numpy as np

# FER Plus coded label
fer_plus_code = {
    'contempt': 7,
    'neutral': 6,
    'sad': 4,
    'happy': 3,
    'surprise': 5,
    'angry': 0,
    'fear': 2,
    'disgust': 1
}

df1 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/kb_0%.csv")
print(f"labels: {sorted(df1['true_label'].unique().tolist())}")
df2 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/private_test_0%.csv")
print(f"labels: {sorted(df2['true_label'].unique().tolist())}")
df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/fer_plus_data.csv")
print(f"original FER: {df.columns.tolist()}")