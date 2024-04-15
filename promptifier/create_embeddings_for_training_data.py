import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the pre-trained Sentence Transformer model (e.g., 'bert-base-nli-mean-tokens')
model = SentenceTransformer('sentence-t5-base')

def embed_df(path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(path, encoding='latin-1')
    if "rewrite_prompt_v" in df.columns:
        print("Embeddings already present in csv")
        return


    # Apply the Sentence Transformer model to generate embeddings for each rewrite prompt
    columns_to_embedd = ['rewrite_prompt']
    for col in columns_to_embedd:
        embeddings = model.encode(df[col].tolist(), show_progress_bar=True)
        df[col+'_v'] = [emb.tolist() for emb in embeddings]

    # Save the DataFrame to a new CSV file with the embeddings
    df.to_csv(path, index=False)