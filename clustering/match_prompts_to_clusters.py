
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
tqdm.pandas()
# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('sentence-transformers/sentence-t5-base')



source_folder = './load_by_class'
directory = './load_by_class'  # Replace with your directory path
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
print(csv_files)

def score(submission, test):
    
    scs = lambda row: abs((cosine_similarity(row["actual_embeddings"], row["pred_embeddings"])) ** 3)
    

    submission["actual_embeddings"] = test["rewrite_prompt"].progress_apply(lambda x: embedding_model.encode(x, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1))
    submission["pred_embeddings"] = submission["rewrite_prompt"].progress_apply(lambda x: embedding_model.encode(x, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1))
    
    submission["score"] = submission.apply(scs, axis=1)
    
    return np.mean(submission['score'])[0][0]


def get_prompt(row):
    return "hi"

prompt = "hi"
for file in csv_files:
    print(file)
    data_df = pd.read_csv(os.path.join(directory, file))
    submission_df = data_df
    submission_df["rewrite_prompt"] = data_df.apply(get_prompt)
    print(data_df["rewrite_prompt"])

    print(data_df["rewrite_prompt"])
    score(submission_df,data_df)