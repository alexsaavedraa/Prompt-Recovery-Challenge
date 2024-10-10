# Reverse Prompt Engineering with topic-specific Adversarial Prompts

## 1. Introduction

The Reverse Prompt Engineering Competition is a machine learning competition where competitors, given a sample of original text, and that text rewritten by an LLM, For example, if given a text on the history of trees, and a sonnet about trees, your code might  predict "Rewrite this text in the style of Shakespeare"




## 2. Data Generation
### Bigrams
The first step in the process was to generate data. In order to create a dataset that was both highly diverse, and highly representative of the competition dataset, we iteratively generated datasets from a bigram generator, with a source text of 10k bigrams, then filtered out the data that was not representative of the competition data (as measured by cosine similarity scores to sentences with known scores). The filter ultimately screened out about 99.5% of the data generated, but due to the speed of generation (roughly 10k sentences/minute), we were still able to quickly generate large and high diverse representative datasets. 

### Quantifying Data Quality
We used two different quantitative measures of data quality: diversity, and representativeness. To measure diversity, we created an adjacency matrix of cosine similarities between data points, and calculated basic statistics on them, with median being most important. The figure below shows how the diversity changes for each generation of data. For generations 2-10, as shown, the diversity score is getting worse, because the screen for representativeness was biased towards some types of sentences more than others. This is because representativeness is calculated by finding the average cosine similarity across many known quantities; however, the known quantities were only painting a very partial picture of the actual data. It’s like knowing that the dataset definitely contains sentences about scat music  (which it did), so you become biased towards generating more data related to scat music. However, we were able to correct this in later generations by biasing against weighting too heavily in favor of known topics. Generations 11-15 subsequently get better diversity scores. 

![alt text](<Screenshot 2024-06-20 104641-1.png>)




### Why Use Bigrams? + One More Data Trick 
If you know anything about bigram based predictive text generation, you know its one level removed from random text generation. Our bigram text datasets were full of nonsense prompts like
``` Modify please craft it marked as a series of open as if it into a grumpy old wise trees advice they were```
But in our case, that is a feature, not a bug. During the competition, a number of people publicly released their own synthetic datasets. Most of these datasets were generated with ChatGPT or some other LLM. Using an LLM creates universe sentences. You can usually tell how some piece of text is made by ChatGPT, right? Well that writing style uses a limited vocabulary and number of topics. The same is true of many different types text generators to varying degrees. By using a bigram method with a sufficiently large dataset, we were able to generate strings of words that were close to random. This was a boon for textual diversity, but terrible for having sentences make sense. For an adversarial attack, thankfully, making sense was not a priority. However, for testing, we wanted to ensure our dataset was as close to the competition as possible.  We found that we could pass our nonsense bigram prompts to ChatGPT and instruct it to rewrite the prompt in a way that makes sense.  For example, the prompt for earlier became  ```Transform the text into a series of wise advice from a grumpy old tree.```  We used this method created our final high quality datasets (v12 and v15) that contained (arguably) human readable text that preserved a diversity of topics and vocabulary that represented the competition datasets.


## 3. Mean Prompt Search with Genetic Algorithim 
### The Basic Algorithim

After data generation, we used a genetic algorithm, affectionately named Promptifyer 4, in order to optimize a prompt with the highest cosine similarity to the dataset as a whole. The optimization was completed with a genetic greedy hill climb algorithm. In each generation, the parents are mutated in various ways to produce children. Usually about 600 children per parent and 2-5 parents. As you can imagine, there was a lot of data to keep track of, and without an optimized algorithm, progress was slow. Much of the work in Promptifyer versions 2 and 3 was centered around decreasing the time per generation. This came in the form of CUDA optimizations, multithreading, better data management, and faster scoring metrics. From the original, completely naive implementation, we were able to cut down generation time from about six minutes to 80 seconds, or a 4 fold increase in speed. With these optimizations made, we could run many more generations to find really high-quality mean prompt.

Below you can see the results of several different runs graphed as a generation vs predicted score. Interestingly, you can see that regardless of seed, the shape remains constant for each generation, of roughly a log increasing score.


![alt text](<Score vs generation many runs-1.png>)


### Flatlining scores
Because the shape is constant, we knew we could predict early on which starting points would perform well or poorly after many many generations, and as illustrated by the figure below, our predictions remain accurate in small and large numbers of generations:


![alt text](<score vs generation with predictionspng-1.png>)


### Eeking out a winning score with Genetic Pruning
With this information, we know that our best score is limited by how good of a starting seed we have and how much compute time we have, with current methods. An examination of the most fit prompts we generated, we found that they were extremely long (50 to 60 words) and that their score leveled off drastically once they got to that range. In order to combat this, we implemented a genetic pruning algorithm, which allowed us to shorted the prompt without losing too much score, and avoid being stuck in local maximum. This addition really allowed us to eek every last bit of cosine similarity activation out of each prompt. You can see the pruning very visibly in the graph below, which only took place after the score increase flatlined for many generations. Each successive pruning basically "rerolled" the seed we were using, with the final pruning, in orange being the most extreme by a large margin.




![alt text](<score vs generation with debloater-1.png>)




With Promptifyer 4 complete, with all of its features and optimizations, we had developed a form of adversarial attack against the scoring system. An adversarial attack relies on a string of tokens that achieves an unusually high activation from sentence t-5 base. Our winning submission uses one such string of tokens for every single prompt. However, we theorized that we could divide prompts into categories, and have a tailor-made attack prompt for each category, as seen in the next section.



## 4. Topic Classification

The above strategy describes how we found our best performing prompts, but we believed that we could do better. What if we could separate semantically similar prompts into clusters, and submit a mean prompt tailor made for that cluster?  Theoretically we should be able to achieve much higher cosine similarity score on each individual cluster with this method, and we already had a strong way of finding mean prompts on data. However, in order to prove that this would be possible, we need to first show that semantic clusters exist. A U-map reduction with a k-means clustering is shown below:

![alt text](<Umap Projection with K means clustering-1.png>)

We indeed were able to cluster prompts based on semantic meaning, but several challenges remained. First, we had to decide on an optimal number of clusters and how to form them. We ultimately settled on training a BERTopic model on our dataset, and clustered based on a k-means method on a UMAP reduction to 2 dimensions. Then, using BERTopic, we were able to identify the topics of each cluster. A minimum cluster size of 100 left us with 13 clusters that we were able to find cluster prompts for (in the cluster prompts section of the code below), using Promptifyer 4. These prompts performed from 1% to 8% better on prompts in their cluster, and 1% to 8% worse on prompts from outside of their cluster.  

The cluster specific performance makes it critical that we are able to accurately classify which cluster a prompt belongs to, but without seeing the prompt, this becomes very difficult. We used our trained BERTopic model to generate topic vectors for the original and rewritten text, then performed vector subtraction, and matched that to prompt clusters. This method achieved an accuracy of roughly 40%, compared to random guessing at 7% accuracy. Unfortunately, we needed 60% accuracy before we saw meaningful score gains.

## Results

Our results speak for themself. 12th place on the public leaderboard, and 13th place on the private leaderboard, out of 2175 teams.

![alt text](<12th place-1.png>)

Our score dropped about half a percentage point in the private leaderboard. Such a small drop is within natural variations from dataset to dataset, and is evidence that our submissions were not highly overfit.

## Potential Improvements
One avenue for improvement that likely would have pushed us into the top 10 would have been to expand Promptifyer's available vocabulary, which we limited for performance reasons. Sentence-t5-base has a token vocabulary of  32,128, however we only explored whole word tokens from a set of common words, which diminished our potential to discover higher activation prompts. Additionally, there is room to improve our topic prediction methods by limiting our clusters to be more significant, or using a more powerful model than BERTopic.
