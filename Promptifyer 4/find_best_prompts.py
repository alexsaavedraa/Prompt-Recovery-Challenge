import pandas as pd
import numpy as np
import time
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import defaultdict
import re
import string
import random
from itertools import permutations




testing_file = './generations/OldBigrams_BEST.csv'
reference_dataset = './source_data/sorted_data_v13.csv'
origin = 'Promptifyer 3 by Alex Saavedra'
num_rows_to_sample = 200 #measuring cos sim scales linearly with this factor
reference_data_cutoff = 200 # How many of the top sentences to consider
num_top_rows_to_test = 400

df = pd.read_csv(testing_file)
sorted_df = df.sort_values(by='score', ascending=False)
sorted_df = sorted_df[:num_top_rows_to_test]
sorted_df = sorted_df.reset_index(drop=True)
# Print the sorted DataFrame
print(sorted_df.head()['score'])


#GET SOME SCORING DATA
embeddings = pd.read_csv(reference_dataset, encoding = 'latin-1')
embeddings = embeddings[:reference_data_cutoff]
embeddings['rewrite_prompt_v'] = embeddings['rewrite_prompt_v'].apply(lambda x: np.array(ast.literal_eval(x)))

model = SentenceTransformer('sentence-t5-base')

#HEPER FUNCTIONS TO VALIDATE NEW SENTENCES
def get_sample(embeddings_v):
    '''Get a sample from the training data to score against'''
    random_indices = np.random.randint(0, len(embeddings_v), num_rows_to_sample)
    random_sampled_rows = embeddings_v.iloc[random_indices]['rewrite_prompt_v']
    return random_sampled_rows
def predict_score(sentence_v):
    '''accepts a sentence vector, automatically samples rows, and outputs a score'''
    sampled_rows = get_sample(embeddings)
    scores = []
    for embedding in sampled_rows:
            score = abs((cosine_similarity(embedding.reshape(1, -1), np.array(sentence_v).reshape(1, -1))) ** 3)
            scores.append(score)
    return np.array(scores).mean()

def get_all_scores(sentences_df):
    sentences_df['score'] = sentences_df['sentence_v'].apply(predict_score)
    return sentences_df

def make_sentence_vectors_df(sentences_df):
    test_df_sentences_v = model.encode(sentences_df['sentence'], normalize_embeddings=True, show_progress_bar=True)
    sentences_df['sentence_v'] = test_df_sentences_v.tolist()  # Convert to list for DataFrame assignment
    return sentences_df


sentence_df = make_sentence_vectors_df(sorted_df)
sentence_df['genetic_score'] = sentence_df['score']
sentence_df = get_all_scores(sentence_df)
print(sentence_df)     
new_filename = f'results/run_{os.path.basename(testing_file).removesuffix(".csv")}_against_{os.path.basename(reference_dataset).removesuffix(".csv")}.csv'
print(new_filename)
sentence_df = sentence_df[['score', 'generation', 'genetic_score', 'sentence' ]]
sentence_df = sentence_df.sort_values(by='score', ascending=False)
sentence_df.to_csv(new_filename, index=False)