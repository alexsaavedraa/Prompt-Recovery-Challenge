#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
from openai import OpenAI
client = OpenAI(api_key="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")
def get_new_prompt(old_prompt):
    
    completion = client.chat.completions.create(
    
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a super intelligent AI. I am going to give you a garbled llm prompt, and you must tell me the prompt i meant to say. Each prompt will be some instruction to transform or rewrite a piece of text. Change as little as possible to make it meaningful"},
        {"role": "user", "content": old_prompt}
    ]
    )
    return completion.choices[0].message.content
#print(completion.choices[0].message.content)
#print(completion)'

import pandas as pd
import string

df = pd.read_csv('./source_data/original_and_rewrittten.csv', encoding='latin-1')
def has_non_alphanumeric(row):
    for char in row:
        if char not in string.ascii_letters and char not in string.digits and char not in string.punctuation and char != " ":
            return True
    return False
def string_not_empty(row):
     return len(row) > 0 

# Apply the function to the 'Text' column and filter rows
df = df[df['rewrite_prompt'].apply(lambda x: not string_not_empty(x))]
print(df.head(3))
import time
import csv

with open('./gpt_bigram_v4.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            
            for index, row in df.iterrows():
                new_prompt = get_new_prompt(row['rewrite_prompt_old'])
                row['rewrite_prompt'] = new_prompt
                row['file'] = 'bigram_mean_v2'
                writer.writerow(row.to_list())
                time.sleep(0.15)
                print(row["rewrite_prompt"])
