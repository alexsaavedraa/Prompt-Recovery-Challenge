from calculate_diversity import diversity_score
from sentence_transformers import SentenceTransformer
import pandas as pd
import ast
from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import warnings 
import os

warnings.filterwarnings('ignore') 

model = SentenceTransformer('sentence-transformers/sentence-t5-base')

# Read in Data
file = "source_data/gpt_bigrams_v15.csv"
print("Reading", file)
data_df = pd.read_csv(file, encoding='latin-1').head(500)
if "rewrite_prompt_v" in data_df.columns:
    data_df["rewrite_prompt_v"] = data_df["rewrite_prompt_v"].apply(ast.literal_eval)
else:
    data_df["rewrite_prompt_v"] = data_df["rewrite_prompt"].apply(model.encode)
    data_df.to_csv("gpt_bigrams_v15_top_500.csv", encoding='latin-1')

emb = data_df["rewrite_prompt_v"].tolist()

# Cluster on AgglomerativeClustering
print("Creating UMAP Model")
umap_model = UMAP(n_neighbors=7, min_dist=0, metric="cosine", n_components=2, random_state=42)
u = umap_model.fit_transform(emb)
print(len(emb))

print("Creating Clustering Model")
model = AgglomerativeClustering(n_clusters=15)
preds = model.fit_predict(u)
plt.scatter(u[:,0], u[:,1], c=preds)

data_df["class"] = preds
print()

print("Extracting Keywords:")
files = []
for i in set(preds):
    cluster_data = data_df[data_df["class"] == i]
    vectorizer_model = CountVectorizer(stop_words="english")
    X = vectorizer_model.fit_transform(cluster_data["rewrite_prompt"])
    vocab = vectorizer_model.get_feature_names_out()
    word_counts = X.sum(axis=0)
    sorted_word_counts = sorted([(word, word_counts[0, idx]) for word, idx in vectorizer_model.vocabulary_.items() if word not in ["text", "transform", "rewrite", "edit", "revise"]], key=lambda x: x[1], reverse=True)
    name = f"{i}_" + "_".join([x[0] for x in sorted_word_counts[:5]]) + f"_v{i+16}.csv"
    cluster_data.to_csv(os.path.join("diversity_score_data", name))
    print(i, len(cluster_data), sorted_word_counts[:5])
    files.append(name)

print()
data_df.to_csv("results/gpt_bigrams_v15_500_with_class.csv")
scores = diversity_score("results/gpt_bigrams_v15_500.png")
print()

for i, name in enumerate(files):
    print(f"{name} \t\t{scores[i+2]}")




