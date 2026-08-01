import pandas as pd
import glob
import os
import re
from Thesis.VLMs.LLaVa.llava_rag.vis_results import hist_visualization



# getting all file paths in train_test_sets, and visualize class distribution
for path in glob.glob("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/*.csv"):
    df = pd.read_csv(path)
    raw_title = os.path.splitext(os.path.basename(path))[0]
    raw_title = raw_title.replace("[", "").replace("]", "")
    raw_title = re.sub(r"(\d+%)", r"threshold \1", raw_title)
    title = raw_title.replace("_", " ")
    plot = hist_visualization(df, title)
    plot.show()


# jaccard for  KB
kb0 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/kb_0%.csv")
kb50 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/kb_50%.csv")
set_kb0 = set(kb0['file_path'])
set_kb50 = set(kb50['file_path'])
jaccard_similarity = len(set_kb0 & set_kb50) / len(set_kb0 | set_kb50)
jaccard_distance = 1 - jaccard_similarity
print("Jaccard distance KB:", jaccard_distance)


# jaccard for Public test
public_test0 = pd.read_csv('/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/public_test_0%.csv')
public_test50 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/rag_thresholds/train_test_sets/public_test_50%.csv")
set_kb0 = set(public_test0['file_path'])
set_kb50 = set(public_test50['file_path'])
jaccard_similarity = len(set_kb0 & set_kb50) / len(set_kb0 | set_kb50)
jaccard_distance = 1 - jaccard_similarity
print("Jaccard distance Public test:", jaccard_distance)

