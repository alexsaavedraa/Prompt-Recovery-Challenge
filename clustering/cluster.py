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
from bertopic import BERTopic
import pickle

warnings.filterwarnings('ignore') 


# Read in Data
file = "source_data/gpt_bigrams_v15.csv"
print("Reading", file)
data_df = pd.read_csv(file, encoding='latin-1').head(500)
if "rewrite_prompt_v" in data_df.columns:
    data_df["rewrite_prompt_v"] = data_df["rewrite_prompt_v"].apply(ast.literal_eval)
# else:
#     print("Encoding training data")
#     data_df["rewrite_prompt_v"] = data_df["rewrite_prompt"].apply(model.encode)
#     data_df.to_csv("gpt_bigrams_v15_top_500.csv", encoding='latin-1')

emb = data_df["rewrite_prompt_v"].tolist()

# Cluster on AgglomerativeClustering
def agglomerativeCluster():
    print("Creating UMAP Model...")
    umap_model = UMAP(n_neighbors=7, min_dist=0, metric="cosine", n_components=2, random_state=42)
    u = umap_model.fit_transform(emb)
    print(len(emb))

    print("Creating Clustering Model...")
    model = AgglomerativeClustering(n_clusters=15)
    preds = model.fit_predict(u)
    plt.scatter(u[:,0], u[:,1], c=preds)
    return preds

# def berTopic():
print("Starting BERTopic Model...")
embedding_model = SentenceTransformer('sentence-transformers/sentence-t5-base')
vectorizer_model = CountVectorizer(stop_words="english")
umap_model = UMAP(n_neighbors=7, min_dist=0, metric="cosine", n_components=2)
model = BERTopic(
    embedding_model=embedding_model, 
    umap_model=umap_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True
    )
preds, probs = model.fit_transform(data_df["rewrite_prompt"])

data_df["class"] = preds

# DIVERSITY FROM FILE
# data_df = pd.read_csv("results/gpt_bigrams_v15_500_with_class.csv")
# model = pickle.load(open('model.pkl','rb'))


print()

print("Extracting Keywords:")
files = []
for i in set(data_df["class"]):
    cluster_data = data_df[data_df["class"] == i]
    # vectorizer_model = CountVectorizer(stop_words="english")
    # X = vectorizer_model.fit_transform(cluster_data["rewrite_prompt"])
    # vocab = vectorizer_model.get_feature_names_out()
    # word_counts = X.sum(axis=0)
    # sorted_word_counts = sorted([(word, word_counts[0, idx]) for word, idx in vectorizer_model.vocabulary_.items() if word not in ["text", "transform", "rewrite", "edit", "revise"]], key=lambda x: x[1], reverse=True)
    # name = f"{i}_" + "_".join([x[0] for x in sorted_word_counts[:5]]) + f"_v{i+16}.csv"
    name = model.topic_labels_[i]+ f"_v{i+17}.csv"
    cluster_data.to_csv(os.path.join("diversity_score_data", name))
    print(i, len(cluster_data), model.topic_representations_[i][:4])
    files.append(name)

print()
data_df.to_csv("results/berttopic/pickled_default_embed_gpt_bigrams_v15_500_with_class.csv")
scores = diversity_score("results/berttopic/pickled_default_embed_gpt_bigrams_v15_500.png")
print()

for i, name in enumerate(files):
    print(f"{name} \t\t\t {scores[i+2]}")

print()

pickle.dump(model, open('results/berttopic/default_embed_gpt_bigrams_v15_50.pkl','wb'))


