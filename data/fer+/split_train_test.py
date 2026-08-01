# from Thesis.VLMs.LLaVa.llava_rag.create_knowledge_base import create_knowledge_base
import pandas as pd
from Thesis.VLMs.LLaVa.llava_rag.vis_results import hist_visualization


# # use the cleaned data from retinaface using confidence  0.6
# data_to_split = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/files_preprocess/ferplus_cleaned_conf6.csv")
#
# # split the data using a method
# save_to = "/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets"
# test_data = create_knowledge_base(sizes=[0.05,0.1], data_name="fer_plus", full_df=data_to_split,
#                       sampling_method='random', save_dir=save_to)
#
#
# # distribution
# plt_hist = hist_visualization(df=test_data,plot_title='Test set Class Distribution')
# plt_hist.show()
#
# # 5% class distribution
# kb5_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.05.csv")
# plt_hist5 = hist_visualization(df=kb5_df,plot_title='KB(5%) Class Distribution')
# plt_hist5.show()
#
# # 10% class distribution
# kb10_df = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.1.csv")
# plt_hist10 = hist_visualization(df=kb10_df,plot_title='KB(10%) Class Distribution')
# plt_hist10.show()

kb5 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.05.csv")
kb10 = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_kb_0.1.csv")
test = pd.read_csv("/gpfs0/bgu-vilenchi/users/sdolev/Thesis/VLMs/data/fer+/rag_sets/fer_plus_test.csv")
plt_hist5 = hist_visualization(df=kb5,plot_title='KB(5%) Class Distribution')
plt_hist5.show()
plt_hist10 = hist_visualization(df=kb10,plot_title='KB(10%) Class Distribution')
plt_hist10.show()
plt_test = hist_visualization(df=test,plot_title='Test Class Distribution')
plt_test.show()

print(f"kb 5: {len(kb5)}")
print(f"kb 10: {len(kb10)}")
print(f"test: {len(test)}")

print(f"kb 5 out of test: {round(len(kb5) / len(test) * 100,3)}%")
print(f"kb10 out of test: {round(len(kb10)/ len(test) * 100,3)}%")


print(f"kb 5 out of the whole dataste: {round(len(kb5) / (len(test) + len(kb10)),3)}%")
print(f"kb10 out of the whole dataset: {round(len(kb10)/ (len(test) + len(kb10)),3)}%")