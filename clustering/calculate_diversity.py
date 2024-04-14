from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize SentenceTransformer model
model = SentenceTransformer('sentence-transformers/sentence-t5-base')

# Function to calculate pairwise cosine similarity
def calc_cosine_similarity(embeddings):
    similarities = cosine_similarity(embeddings)
    similarities = np.power(similarities, 3)
    return similarities

# Get all CSV files in a directory
directory = './source_data'  # Replace with your directory path
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Extract version numbers from file names and create a color map
version_numbers = [int(file.split('_v')[-1][:-4]) for file in csv_files]
#files_ves = zip(version_numbers, csv_files)
norm = plt.Normalize(min(version_numbers), max(version_numbers))
cmap = plt.get_cmap('viridis')  # You can choose any colormap you like
colors = [cmap(norm(version)) for version in version_numbers]
ls = [] 
for x,y in zip(version_numbers, csv_files):
    #print(x,y)
    ls.append([x,y])
sorted_list = sorted(ls, key=lambda x: x[0])
print(sorted_list)
# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Initialize lists to store cosine similarity values
cosine_similarity_values = []
file_labels = []

# Loop through each CSV file
for idx, file_name in enumerate(sorted_list):
    file_name = file_name[1]
    print("READING FILE ", file_name)
    try:
        
        df = pd.read_csv(os.path.join(directory, file_name), encoding='latin-1')
        prompts = df['rewrite_prompt'].head(500).tolist()  # Extract first 25 sentences
        embeddings = model.encode(prompts, normalize_embeddings=True)
        similarities = calc_cosine_similarity(embeddings)
        similarities_flat = similarities[np.triu_indices(similarities.shape[0], k=1)]
        
        # Add cosine similarity values and corresponding file labels to lists
        cosine_similarity_values.extend(similarities_flat)
        file_labels.extend([file_name[:-4]] * len(similarities_flat))  # Repeat file name for each value
        #print(file_labels)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Create DataFrame from the lists
data = pd.DataFrame({'File': file_labels, 'Cosine Similarity': cosine_similarity_values})

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot box and whisker plot for cosine similarity values with colored boxes based on version numbers
sns.boxplot(y='File', x='Cosine Similarity', data=data, ax=ax, palette=colors)

# Add labels and title
plt.title("Cosine Similarity Box and Whisker Plot")
plt.xlabel("Cosine Similarity")
plt.ylabel("File")

# Show plot
plt.tight_layout()
plt.show()
