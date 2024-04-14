from sentence_transformers.util import pytorch_cos_sim
import numpy as np
import random

def get_sample(embeddings_v, num_rows_to_sample):
        '''Get a sample from the training data to score against'''
        available_indices = set(range(len(embeddings_v)))
        random_indices = []
        for _ in range(num_rows_to_sample):
            selected_index = random.choice(list(available_indices))
            random_indices.append(selected_index)
            available_indices.remove(selected_index)

        random_sampled_rows = embeddings_v.iloc[random_indices]['rewrite_prompt_v']
        return random_sampled_rows

def predict_score(sentence_v, embeddings=None, num_rows_to_sample=None):
    '''accepts a sentence vector, automatically samples rows, and outputs a score'''
    sampled_rows = get_sample(embeddings, num_rows_to_sample)
    scores = []
    for embedding in sampled_rows:
            score = abs((pytorch_cos_sim(embedding.reshape(1, -1), np.array(sentence_v).reshape(1, -1))) ** 3)
            scores.append(score)
    return np.array(scores).mean()

def get_all_scores(sentences_df, embeddings, num_rows_to_sample):
    #print(f"the len of the sub df = {len(sentences_df)} ")
    sentences_df['score'] = sentences_df['sentence_v'].apply(predict_score, embeddings=embeddings, num_rows_to_sample=num_rows_to_sample)
    return sentences_df
