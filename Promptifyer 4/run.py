# import pandas as pd
# import random
# default_color = (random.random(),random.random(),random.random())
# path = ".\\generations\\TransformImprove2_BEST.csv"
# df = pd.read_csv(path)
# df["color"] = [default_color] * len(df)
# df = df[["generation","score","sentence","color"]]
# df.to_csv(path, header=True, index=False)

from utils import sentence_to_list
import pandas as pd
import matplotlib.pyplot as plt
from math import comb

file = "generations\OldBigramsDebloat5_BEST.csv"
df = pd.read_csv(file)
plt.scatter( df["generation"], df["score"]/df["sentence"].apply(lambda x: len(sentence_to_list(x))))
plt.title("sentence length vs. score")
plt.show()


