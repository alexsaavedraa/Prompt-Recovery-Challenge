if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import time
    import ast
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers.util import pytorch_cos_sim
    import os
    from collections import defaultdict
    import re
    import string
    import random
    from itertools import permutations
    import cosSim
    import multiprocessing
    from create_embeddings_for_training_data import embed_df
    ##TODO
    #-add parent mixing via vec2text or some other method
    #-add fun plot of progress over time
    import warnings
    import pandas as pd

    # Suppress FutureWarning messages
    warnings.simplefilter(action='ignore', category=FutureWarning)

    ###PARAMETERS
    #scoring params
    training_file = './source_data/gpt_bigrams_v15.csv'
    origin = 'Promptifyer 3 by Alex Saavedra'
    num_rows_to_sample =  100 #'///// #measuring cos sim scales linearly with this factor
    scorring_data_cutoff = 100 # How many of the top sentences to consider
    #bigram params
    bigram_source_folder = './bigrams/' #the bigrams are made from all .txt files
    all_scored_sentences = './generations/promptifyer_3_Transform_dump.csv'
    best_of_each_gen = './generations/promptifyer_3_Transform_BEST.csv'
    #hyper params
    starting_sentence = "Transform this piece mellor by text so explicitly so amah nairs of or an( enhance thus allowing a the a spouted the hollings intended you wanna get graeae simple be revamp this it one notable and thasos, where a he carters"
    normal_survivors_per_generation = 5
    normal_children_from_bigram_insertion = 300 #linear time
    normal_children_from_bigram_injection = 300 #linear time
    normal_children_from_onegram_injection = 300
    normal_children_from_onegram_insertion = 300
    normal_word_shift_max_distance = 3 #exponential time.  13 words@4 adds 189 variation 6@ 4 = 63, 

    #other parameters
    model = SentenceTransformer('sentence-t5-base')
    all_sentences_tried = set()

    #GET SOME SCORING DATA
    embed_df(training_file)
    embeddings = pd.read_csv(training_file, encoding = 'latin-1')
    embeddings = embeddings[:scorring_data_cutoff]
    embeddings['rewrite_prompt_v'] = embeddings['rewrite_prompt_v'].apply(lambda x: np.array(ast.literal_eval(x)))
    all_words_seen = set()
    all_words_seen.add("first")

    #LOAD BIGRAMS
    def load_bi_grams_from_files(folder_path):
        bi_grams = defaultdict(list)
        global all_words_seen
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        for file_name in files:
            print("loading ", file_name)
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                words = file.read().lower().split()
                for i in range(len(words) - 1):
                    words[i] = words[i].replace('"', '').replace('.','')
                    sw = words[i].replace('"', '').replace('.','').replace('?', '').replace(',','')
                    all_words_seen.add(words[i])
                    bi_gram = (words[i], words[i + 1])
                    bi_grams[bi_gram[0]].append(bi_gram[1])
        all_words_seen = list(all_words_seen)
        return bi_grams
    bigrams = load_bi_grams_from_files(bigram_source_folder)
    print(f"there are {len(bigrams)} bigrams loaded, and {len(all_words_seen)} words seen.")


    #SENTENCE HELPERS
    def sentence_to_list(sentence):
        sentence = sentence.lower()
        tokens = re.findall(r'\w+|\W', sentence)
        filtered_list = [s for s in tokens if s.strip() != ""]
        return filtered_list

        
    def list_to_sentence(sentence_list):
        sentence = ""
        for word in sentence_list:
            if word in string.punctuation:
                sentence += word
            else:
                sentence += ' ' + word
        return sentence

    def trim_and_capitalize_first(sentence):
        first_sentence = re.split(r'(?<=[.!?"]) +', sentence)[0]
        capitalized_sentence = first_sentence.strip().capitalize()
        return capitalized_sentence.replace("'", '').replace('"','')





    #BIGRAM INJECTOR FUNCTION
    def bigram_injection(sentence, num_injections):
        '''Makes specified number of sentences where each
        has one word replaced with a bigram containing that word'''
        try:
            children = set()
            original = sentence_to_list(sentence)
            lst = sentence_to_list(sentence)
            words_to_choose = []
            word_choices = {}
            for word in lst:
                bg_list = bigrams[word]
                if len(bg_list) > 0:
                    words_to_choose.append(word)
                    word_choices[word] = list(set(bg_list))
        
            while len(children) < num_injections:
                lst = original.copy()
                if len(words_to_choose) < 1:
                    break
                inject_point = random.choice(words_to_choose)
                word_choices_list = word_choices[inject_point]
                inject_word = random.choice(word_choices_list)
                word_choices_list.remove(inject_word)
                word_choices[inject_point] = word_choices_list
                if len(word_choices_list) <= 1:
                    words_to_choose.remove(inject_point)
                lst_copy = lst.copy()
                for i, word in enumerate(lst_copy):
                    if word == inject_point:
                        lst.insert(i+1, inject_word)
                child = list_to_sentence(lst)
                child = trim_and_capitalize_first(child)
                children.add(child)
            return(list(children))    
        except:
            return []



    #Bigram INSERTION FUNCTION
    def bigram_insertion(sentence, num_insertions):
        try:
            children = set()
            original = sentence_to_list(sentence)
            lst = sentence_to_list(sentence)
            insertions_attempted = 0
            while insertions_attempted < num_insertions:
                lst = original.copy()
                insertions_attempted += 1
                bigram_to_insert_start = random.choice(list(bigrams.keys()))
                bigram_to_insert_end = random.choice(bigrams[bigram_to_insert_start])
                location = random.choice(list(range(len(lst)+1)))
                lst.insert(location, bigram_to_insert_end)
                lst.insert(location, bigram_to_insert_start)
                res = trim_and_capitalize_first(list_to_sentence(lst))
                children.add(res)
            return list(children)
        except:
            return []

    def make_all_single_word_removals(sentence):
        '''returns a list of all possible variations of the original sentence with 1 word removed'''
        children = set()
        original = sentence_to_list(sentence)
        lst = sentence_to_list(sentence)
        for i in range(len(original)):
            lst = original.copy()
            del lst[i]
            res = trim_and_capitalize_first(list_to_sentence(lst))
            children.add(res)
        return list(children)
            

    def make_variations(sentence):
        lst = sentence_to_list(sentence)
        variations = set()
        for i in range(len(lst)):
            for perm in permutations(lst[i:min(i+word_shift_max_distance, len(lst))]):
                new_sentence = ' '.join(lst[:i] + list(perm) + lst[i+len(perm):])
                res = sentence_to_list(new_sentence)
                res = trim_and_capitalize_first(list_to_sentence(res))
                variations.add(res)
        return list(variations)

    #Onegram INSERTION FUNCTION
    def onegram_insertion(sentence, num_insertions):
        children = set()
        original = sentence_to_list(sentence)
        lst = sentence_to_list(sentence)
        insertions_attempted = 0
        while insertions_attempted < num_insertions:
            lst = original.copy()
            insertions_attempted += 1
            oneram_to_insert = random.choice(all_words_seen)
            location = random.choice(list(range(len(lst)+1)))
            lst.insert(location, oneram_to_insert)
            res = trim_and_capitalize_first(list_to_sentence(lst))
            children.add(res)
        return list(children)


    #Onegram_injection
    def onegram_injection(sentence, num_insertions):
        children = set()
        original = sentence_to_list(sentence)
        lst = sentence_to_list(sentence)
        insertions_attempted = 0
        while insertions_attempted < num_insertions:
            lst = original.copy()
            insertions_attempted += 1
            oneram_to_inject = random.choice(all_words_seen)
            location = random.choice(list(range(len(lst))))
            lst.insert(location, oneram_to_inject)
            del lst[location + 1]
            res = trim_and_capitalize_first(list_to_sentence(lst))
            children.add(res)
        return list(children)








    #HEPER FUNCTIONS TO VALIDATE NEW SENTENCES
    def get_sample(embeddings_v):
        '''Get a sample from the training data to score against'''
        available_indices = set(range(len(embeddings_v)))
        random_indices = []
        for _ in range(num_rows_to_sample):
            selected_index = random.choice(list(available_indices))
            random_indices.append(selected_index)
            available_indices.remove(selected_index)

        random_sampled_rows = embeddings_v.iloc[random_indices]['rewrite_prompt_v']
        return random_sampled_rows
    def predict_score(sentence_v):
        '''accepts a sentence vector, automatically samples rows, and outputs a score'''
        sampled_rows = get_sample(embeddings)
        scores = []
        for embedding in sampled_rows:
                score = abs((pytorch_cos_sim(embedding.reshape(1, -1), np.array(sentence_v).reshape(1, -1))) ** 3)
                scores.append(score)
        return np.array(scores).mean()

    def get_all_scores(sentences_df):
        sentences_df['score'] = sentences_df['sentence_v'].apply(predict_score)
        return sentences_df
    
    def multiprocess_scores(sentences_df):
        #print(f"original df: {sentences_df}" )
        num_cores = multiprocessing.cpu_count()
        #print(f"there are f{num_cores}")
        chunks = np.array_split(sentences_df, num_cores)  # Split dataframe into chunks

        with multiprocessing.Pool(num_cores) as pool:
            processed_chunks = pool.starmap(cosSim.get_all_scores, [(chunk, embeddings, num_rows_to_sample) for chunk in chunks])

        processed_df = pd.concat(processed_chunks)  # Concatenate processed chunks back into a dataframe
        #print(f"multiprocessed df: {processed_df}" )
        return processed_df

    def make_sentence_vectors_df(sentences_df):
        test_df_sentences_v = model.encode(sentences_df['sentence'], normalize_embeddings=True, show_progress_bar=True, convert_to_tensor=True)
        sentences_df['sentence_v'] = test_df_sentences_v.tolist()  # Convert to list for DataFrame assignment
        return sentences_df

    #Simple IO
    def append_to_csv(df, csv_file):
        file_exists = os.path.isfile(csv_file)
        mode = 'a' if file_exists else 'w'
        header = True if not file_exists else False
        df.to_csv(csv_file, mode=mode, header=header, index=False)


    ##Genetic algorithm functions

    def make_children(parent):
        '''accepts a single parent sentence and makes children according to the hyperparameter'''
        new_children = [parent]
        new_children += bigram_injection(parent, children_from_bigram_injection)
        new_children += bigram_insertion(parent, children_from_bigram_insertion)
        new_children += onegram_injection(parent, children_from_onegram_injection)
        new_children + onegram_insertion(parent, children_from_bigram_insertion)
        new_children += make_all_single_word_removals(parent)
        new_children += make_variations(parent)
        return new_children


    def find_fittest(sentences_df):
        df_sorted = sentences_df.sort_values(by='score', ascending=False).head(survivors_per_generation)
        print(df_sorted[['score', 'sentence']])
        res = df_sorted['sentence'].tolist()
        return res, df_sorted


    def advance_to_next_generation(parents_list, generation = 0):
        print(f"Advancing to generation # {generation}")
        global all_sentences_tried
        new_gen = []
        for parent in parents_list:
            new_gen += make_children(parent)
        filtered_generation = [string for string in new_gen if string not in all_sentences_tried]
        for sentence in filtered_generation:
            all_sentences_tried.add(sentence)
        #make dataframe and score it
        filtered_generation += parents_list
        sentence_df = pd.DataFrame({'sentence': filtered_generation, 'generation': generation, 'training_data_used': training_file, 'Origin': origin})
        sentence_df = make_sentence_vectors_df(sentence_df)
        sentence_df = multiprocess_scores(sentence_df)
        sentence_df = sentence_df[['generation', 'score', 'sentence']]
        new_parents_list, new_parents_df = find_fittest(sentence_df)
        #basically, ensure that any duplicate parents get rescored
        for new_parent in new_parents_list:
            for parent in parents_list:
                if parent == new_parent:
                    all_sentences_tried.discard(parent)
                    # print("REMOVING PARENT")
        

        append_to_csv(sentence_df, all_scored_sentences)
        append_to_csv(new_parents_df, best_of_each_gen)
        return new_parents_list
    #advance_to_next_generation([starting_sentence])


    num_generations = 2000
    next_gen = [starting_sentence]
    gen = 0
    for i in range(num_generations*3):
        # if i%3==0:
            survivors_per_generation = normal_survivors_per_generation
            children_from_bigram_insertion = normal_children_from_bigram_insertion
            children_from_bigram_injection = normal_children_from_bigram_injection #linear time
            children_from_onegram_injection = normal_children_from_onegram_injection
            children_from_onegram_insertion = normal_children_from_onegram_insertion
            word_shift_max_distance = normal_word_shift_max_distance #exponential time.  13 words@4 adds 189 variation 6@ 4 = 63, 
            start_of_generation = time.time()
            next_gen = advance_to_next_generation(next_gen, generation=i)
            end_of_generation = time.time()
            print(f"generation {i} took {end_of_generation-start_of_generation} seconds")
        # else:
        #     children_from_bigram_insertion = 0
        #     children_from_bigram_injection = 0 #linear time
        #     children_from_onegram_injection = 0
        #     children_from_onegram_insertion = 0
        #     word_shift_max_distance = 0

        #     next_gen = advance_to_next_generation(next_gen, generation=i)
        #     print(f"generation {i/3} took {time.time()-end_of_generation} seconds")

