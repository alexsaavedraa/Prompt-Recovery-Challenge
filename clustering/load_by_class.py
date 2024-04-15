import pandas as pd
import pickle
import os

data_df = pd.read_csv("results/berttopic/pickled_custom_umap_gpt_bigrams_v15_500_with_class.csv")
model = pickle.load(open('results/berttopic/custom_umap_gpt_bigrams_v15_50.pkl','rb'))

for i in set(data_df["class"]):
    cluster_data = data_df[data_df["class"] == i]
    name = model.topic_labels_[i]+ f"_v{i+17}.csv"
    cluster_data.to_csv(os.path.join("load_by_class", name))
    print(i, len(cluster_data), model.topic_representations_[i][:4])
