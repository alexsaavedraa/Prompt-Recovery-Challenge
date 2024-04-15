import re
import random
import string
from itertools import permutations
import pandas as pd
import os
import matplotlib.pyplot as plt

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

def embed_csv(path, model):
    df = pd.read_csv(path, encoding='latin-1')
    if "rewrite_prompt_v" in df.columns:
        print("Embeddings already present in csv")
        return
    columns_to_embedd = ['rewrite_prompt']
    for col in columns_to_embedd:
        embeddings = model.encode(df[col].tolist(), show_progress_bar=True, device="cuda")
        df[col+'_v'] = [emb.tolist() for emb in embeddings]
    df.to_csv(path, index=False)

def append_to_csv(df, csv_file, do):
    if do:
        file_exists = os.path.isfile(csv_file)
        mode = 'a' if file_exists else 'w'
        header = True if not file_exists else False
        df.to_csv(csv_file, mode=mode, header=header, index=False)

def update_plot(data_path, plot_path, do):
    if do:
        data = pd.read_csv(data_path)
        df = pd.DataFrame(data)

        plt.figure(figsize=(8, 6))
        plt.scatter(df['generation'], df['score'])

        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.title('Score vs Generation')
        plt.savefig(plot_path)

def find_fittest(sentences_df, limit):
    df_sorted = sentences_df.sort_values(by='score', ascending=False)
    df_unique = df_sorted.drop_duplicates(subset=['sentence']).head(limit)
    print(df_unique[['score', 'sentence']])
    res = df_unique['sentence'].tolist()
    return res, df_unique


class ChildrenMaker():
    def __init__(self, bigrams, words_set, word_shift, num_injections, num_insertions):
        self.bigrams = bigrams
        self.all_words_seen = list(words_set)
        self.word_shift = word_shift
        self.num_injections = num_injections
        self.num_insertions=num_insertions

    def make_children(self, parent):
        '''accepts a single parent sentence and makes children according to the hyperparameter'''
        new_children = [parent]
        new_children += self.bigram_injection(parent)
        new_children += self.bigram_insertion(parent)
        new_children += self.onegram_injection(parent)
        new_children + self.onegram_insertion(parent)
        new_children += self.make_all_single_word_removals(parent)
        new_children += self.make_variations(parent)
        return new_children

    def bigram_injection(self, sentence):
        '''Makes specified number of sentences where each
        has one word replaced with a bigram containing that word'''
        try:
            children = set()
            original = sentence_to_list(sentence)
            lst = sentence_to_list(sentence)
            words_to_choose = []
            word_choices = {}
            for word in lst:
                bg_list = self.bigrams[word]
                if len(bg_list) > 0:
                    words_to_choose.append(word)
                    word_choices[word] = list(set(bg_list))
        
            while len(children) < self.num_injections:
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

    def bigram_insertion(self, sentence):
        try:
            children = set()
            original = sentence_to_list(sentence)
            lst = sentence_to_list(sentence)
            insertions_attempted = 0
            while insertions_attempted < self.num_insertions:
                lst = original.copy()
                insertions_attempted += 1
                bigram_to_insert_start = random.choice(list(self.bigrams.keys()))
                bigram_to_insert_end = random.choice(self.bigrams[bigram_to_insert_start])
                location = random.choice(list(range(len(lst)+1)))
                lst.insert(location, bigram_to_insert_end)
                lst.insert(location, bigram_to_insert_start)
                res = trim_and_capitalize_first(list_to_sentence(lst))
                children.add(res)
            return list(children)
        except:
            return []

    def make_all_single_word_removals(self, sentence):
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
            
    def make_variations(self, sentence):
        lst = sentence_to_list(sentence)
        variations = set()
        for i in range(len(lst)):
            for perm in permutations(lst[i:min(i+self.word_shift, len(lst))]):
                new_sentence = ' '.join(lst[:i] + list(perm) + lst[i+len(perm):])
                res = sentence_to_list(new_sentence)
                res = trim_and_capitalize_first(list_to_sentence(res))
                variations.add(res)
        return list(variations)

    def onegram_insertion(self, sentence):
        children = set()
        original = sentence_to_list(sentence)
        lst = sentence_to_list(sentence)
        insertions_attempted = 0
        while insertions_attempted < self.num_insertions:
            lst = original.copy()
            insertions_attempted += 1
            oneram_to_insert = random.choice(self.all_words_seen)
            location = random.choice(list(range(len(lst)+1)))
            lst.insert(location, oneram_to_insert)
            res = trim_and_capitalize_first(list_to_sentence(lst))
            children.add(res)
        return list(children)

    def onegram_injection(self, sentence):
        children = set()
        original = sentence_to_list(sentence)
        insertions_attempted = 0
        while insertions_attempted < self.num_injections:
            lst = original.copy()
            insertions_attempted += 1
            oneram_to_inject = random.choice(self.all_words_seen)
            location = random.choice(list(range(len(lst))))
            lst.insert(location, oneram_to_inject)
            del lst[location + 1]
            res = trim_and_capitalize_first(list_to_sentence(lst))
            children.add(res)
        return list(children)