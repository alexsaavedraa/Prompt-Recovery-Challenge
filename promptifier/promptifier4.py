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
from utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)

class Promptifier():
    def __init__(self, 
                 training_file, 
                 survivors_per_gen=1, 
                 injection_children=50,
                 insertion_children=50,
                 num_rows_to_score_against=100,
                 word_shift_distance=2):
        
        self.origin = 'Promptifyer 4 by Alex Saavedra and Prachi Patil'
        self.max_top_sentence_streak = 10

        self.embedding_model = SentenceTransformer('sentence-t5-base')
        
        self.training_file = training_file
        embed_csv(self.training_file, self.embedding_model)
        
        self.num_rows_to_sample =  num_rows_to_score_against
        self.scoring_data_cutoff = num_rows_to_score_against

        self.bigram_source_folder = './bigrams/' #the bigrams are made from all .txt files
        
        self.survivors_per_generation = survivors_per_gen

        self.all_sentences_tried = set()
        
        embeddings = pd.read_csv(training_file, encoding = 'latin-1')[:num_rows_to_score_against]
        embeddings['rewrite_prompt_v'] = embeddings['rewrite_prompt_v'].apply(lambda x: np.array(ast.literal_eval(x)))
        self.embeddings_df = embeddings
        
        self.all_words_seen = set()
        self.bigrams = self.load_bi_grams_from_files()
        self.child_maker = ChildrenMaker(self.bigrams, self.all_words_seen, word_shift_distance, injection_children, insertion_children)

    def load_bi_grams_from_files(self):
        bi_grams = defaultdict(list)
        files = [f for f in os.listdir(self.bigram_source_folder) if f.endswith('.txt')]
        for file_name in files:
            print("loading ", file_name)
            with open(os.path.join(self.bigram_source_folder, file_name), 'r', encoding='utf-8') as file:
                words = file.read().lower().split()
                for i in range(len(words) - 1):
                    words[i] = words[i].replace('"', '').replace('.','')
                    self.all_words_seen.add(words[i])
                    bi_gram = (words[i], words[i + 1])
                    bi_grams[bi_gram[0]].append(bi_gram[1])
        print(f"there are {len(bi_grams)} bigrams loaded, and {len(self.all_words_seen)} words seen.")
        return bi_grams

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
        print(f"Advancing to generation # {generation}")
        new_gen = []
        for parent in self.parents:
            new_gen += self.child_maker.make_children(parent)
        filtered_generation = [string for string in new_gen if string not in self.all_sentences_tried]
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
        new_parents_list, new_parents_df = find_fittest(sentence_df, self.survivors_per_generation)
        
        curr_top_sentence = new_parents_list[0]
        if curr_top_sentence == self.top_sentence:
            self.top_sentence_streak += 1
        else:
            self.top_sentence = curr_top_sentence
            self.top_sentence_streak = 0
        self.top_score =float(new_parents_df["score"].iloc[0])

        if self.top_sentence_streak > self.max_top_sentence_streak:
            new_parents_list.remove(self.top_sentence)
        
        parents_moving_on = list(set(self.parents).intersection(new_parents_list))
        for parent in parents_moving_on:
            self.all_sentences_tried.discard(parent)

        append_to_csv(new_parents_df, self.best_of_each_gen, self.track_progress)
        update_plot(self.best_of_each_gen, self.plot_filepath, self.track_progress)
        self.parents = new_parents_list

    def promptify(self, run_name = "promptifyer_4", 
                  starting_sentence="Rewrite this text.",
                  num_generations=2000,
                  track_progress=True):
        self.parents = [starting_sentence]
        self.top_sentence = starting_sentence
        self.top_sentence_streak =0
        self.track_progress = track_progress

        self.best_of_each_gen = f'./generations/{run_name}_BEST.csv'
        self.plot_filepath = f'./generations/{run_name}_BEST.png'
        for i in range(num_generations):
            start_of_generation = time.time()
            self.advance_to_next_generation(generation=i)
            end_of_generation = time.time()
            print(f"generation {i} took {end_of_generation-start_of_generation} seconds")
            print(f"current top sentence streak: {self.top_sentence_streak}")

if __name__=="__main__":
    starting_sentence = "Transform this piece mellor by text so explicitly so nairs an si of( enhance thus being would one meaning allowing the an the a spouted the hollings intended you wanna get graeae simple be revamp this known at with one notable and ici, if a he carters"
    p = Promptifier(
        training_file="./source_data/gpt_bigrams_v15.csv",
        survivors_per_gen=5, 
        injection_children=300,
        insertion_children=300,
        num_rows_to_score_against=100,
        word_shift_distance=3
        )
    
    p.promptify(run_name="TransformImprove2", starting_sentence=starting_sentence)
    print(f"{p.top_score}: {p.top_sentence}")