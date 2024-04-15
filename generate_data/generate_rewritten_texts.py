#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
from openai import OpenAI
client = OpenAI(api_key="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")
def get_new_prompt(rewrite_prompt, original_text):
    
    completion = client.chat.completions.create(
    
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a super intelligent AI. Please follow the instructions carefully."},
        {"role": "user", "content": rewrite_prompt + " " +  original_text}
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
     if type(row) == type("string"):
        return len(row) > 0 
     return False

# Apply the function to the 'Text' column and filter rows
df = df[df['rewrite_prompt'].apply(lambda x: string_not_empty(x))]
print(df.head(3))
df = df[df['original_text'].apply(lambda x: string_not_empty(x))]
print(df.head(3))
import time
import csv

with open('./gpt_bigram_v4.csv', mode='w', newline='', encoding='latin-1') as file:
            
                writer = csv.writer(file)
                
                for index, row in df.iterrows():
                    try:
                        rewritten_text = get_new_prompt(row['rewrite_prompt'], row['original_text'])
                        row['rewritten_text'] = rewritten_text
                        row['file'] = 'gpt_rewrittens_v15'
                        writer.writerow(row.to_list())
                        time.sleep(0.15)
                        print("\nog", row['original_text'])
                        print("\nrp", row['rewrite_prompt'])
                        print("\nog", row['rewritten_text'])
                    except:
                        print(f"ROW {index} FAILED! Moving on!")


                    
