import pandas as pd
import matplotlib.pyplot as plt

# Example dataframe
data = pd.read_csv('./promptifyer_3_v15_BEST.csv')
df = pd.DataFrame(data)

# Define colors for each generation


# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['generation'], df['score'])

plt.xlabel('Generation')
plt.ylabel('Score')
plt.title('Score vs Generation')
plt.legend()
plt.show()
