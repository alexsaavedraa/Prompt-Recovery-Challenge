import pandas as pd
import numpy as np
import time
import ast
from sentence_transformers import SentenceTransformer
import os
from collections import defaultdict
import cosSim
import multiprocessing
import warnings
import pandas as pd
import random
from utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)

class Debloater():
    def __init__(self, 
                 training_file, 
                 survivors_per_gen=1, 
                 deletion_children=50,
                 num_rows_to_score_against=100,
                 max_top_sentence_streak=4):
        
        self.origin = 'Promptifyer 4 by Alex Saavedra and Prachi Patil'
        self.max_top_sentence_streak = max_top_sentence_streak

        self.embedding_model = SentenceTransformer('sentence-t5-base')
        
        self.training_file = training_file
        embed_csv(self.training_file, self.embedding_model)
        
        self.num_rows_to_sample =  num_rows_to_score_against
        self.scoring_data_cutoff = num_rows_to_score_against
        
        self.survivors_per_generation = survivors_per_gen

        self.all_sentences_tried = set()
        
        embeddings = pd.read_csv(training_file, encoding = 'latin-1')[:num_rows_to_score_against]
        embeddings['rewrite_prompt_v'] = embeddings['rewrite_prompt_v'].apply(lambda x: np.array(ast.literal_eval(x)))
        self.embeddings_df = embeddings

        self.child_maker = WordRemover(num_removals=deletion_children)
        
    def make_sentence_vectors_df(self, sentences_df):
        test_df_sentences_v = self.embedding_model.encode(sentences_df['sentence'], normalize_embeddings=True, show_progress_bar=True, convert_to_tensor=True, device='cuda')
        sentences_df['sentence_v'] = test_df_sentences_v.tolist()  # Convert to list for DataFrame assignment
        return sentences_df
    
    def multiprocess_scores(self, sentences_df):
        num_cores = multiprocessing.cpu_count()
        chunks = np.array_split(sentences_df, num_cores)  # Split dataframe into chunks

        with multiprocessing.Pool(num_cores) as pool:
            processed_chunks = pool.starmap(cosSim.get_all_scores, [(chunk, self.embeddings_df, self.num_rows_to_sample) for chunk in chunks])

        processed_df = pd.concat(processed_chunks)  # Concatenate processed chunks back into a dataframe
        return processed_df
    
    def advance_to_next_generation(self, generation = 0):
        print(f"DEBLOATER: Advancing to generation # {generation}")
        new_gen = []
        for parent in self.parents:
            new_gen += self.child_maker.make_children(parent)
        filtered_generation = list(set([string for string in new_gen if string not in self.all_sentences_tried]))
        for sentence in filtered_generation:
            self.all_sentences_tried.add(sentence)

        filtered_generation += self.parents
        sentence_df = pd.DataFrame({'sentence': filtered_generation, 
                                    'generation': generation, 
                                    'training_data_used': self.training_file, 
                                    'Origin': self.origin})
        sentence_df = self.make_sentence_vectors_df(sentence_df)
        sentence_df = self.multiprocess_scores(sentence_df)
        sentence_df = sentence_df[['generation', 'score', 'sentence']]
        new_parents_list, new_parents_df = find_all_fittest(sentence_df, self.survivors_per_generation)

        curr_top_sentence = new_parents_list[0]
        if curr_top_sentence == self.top_sentence:
            self.top_sentence_streak += 1
        else:
            self.top_sentence = curr_top_sentence
            self.top_sentence_streak = 0
        self.top_score = float(new_parents_df["score"].iloc[0])
        self.top_sentence_len = len(sentence_to_list(self.top_sentence))

        if self.top_sentence_streak >= self.max_top_sentence_streak: # and len(sentence_to_list(self.parents[1])) <= self.top_sentence_len:
            new_parents_list.remove(self.top_sentence)
        print("DEBLOATER: Total children tested is", len(new_parents_list))
        new_parents_list = new_parents_list[:self.survivors_per_generation]
        new_parents_df = new_parents_df.head(self.survivors_per_generation)

        parents_moving_on = list(set(self.parents).intersection(new_parents_list))
        for parent in parents_moving_on:
            self.all_sentences_tried.discard(parent)

        if self.track_progress:
            append_to_csv(new_parents_df, self.best_of_each_gen, self.plot_color)
            update_plot(self.best_of_each_gen, self.plot_filepath)
        self.parents = new_parents_list

    def debloat(self, run_name = "promptifyer_4", 
                  starting_sentence="Rewrite this text.",
                  starting_generation=0,
                  num_generations=2000,
                  track_progress=True):
        self.parents = [starting_sentence]
        self.top_sentence = starting_sentence
        self.top_sentence_streak = 0
        self.top_sentence_len = 10000000
        self.track_progress = track_progress
        self.plot_color = (random.random(),random.random(),random.random())

        self.best_of_each_gen = f'./generations/{run_name}_BEST.csv'
        self.plot_filepath = f'./generations/{run_name}_BEST.png'
        for i in range(starting_generation, num_generations):
            start_of_generation = time.time()
            self.advance_to_next_generation(generation=i)
            end_of_generation = time.time()
            print(f"DEBLOATER: generation {i} took {end_of_generation-start_of_generation} seconds")
            print(f"DEBLOATER: current top sentence streak: {self.top_sentence_streak}")

if __name__=="__main__":
    starting_sentence = "Improve a tone and thine crafted touch of the text as it and with lay the the with an what instead nw do if done for merrily known what reaches of of of of of of place through including a an being the from or now in is of express ya rewrite this"

    p = Debloater(
        training_file="./source_data/gpt_bigrams_v15.csv",
        survivors_per_gen=5, 
        deletion_children=300,
        num_rows_to_score_against=100,
        max_top_sentence_streak=4
        )
    
    p.debloat(run_name="KaggleLeaderboardDebloat2", 
                starting_sentence=starting_sentence,
                starting_generation=0)
    print(f"DEBLOATER: {p.top_score}: {p.top_sentence}")