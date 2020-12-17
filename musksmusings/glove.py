import numpy as np
import os
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd


class gloveEmbedding:
    def __init__(self):
        self.csvFileDir = "tweets.csv"
        self.tweets = self.load_tweets()
        self.tweetGlove = self.glove_tweets()

    def load_tweets(self):
        temp = pd.read_csv(self.csvFileDir)
        tweets = temp["text"]
        return tweets

    def glove_tweets(self):
        tweetGlove = []
        for i in self.tweets:
            temp_glove = glove(i)
            tweetGlove.append((i, temp_glove))
        return tweetGlove


class glove:
    def __init__(self, tweet):
        self.glovePath = "glove.6B.300d.txt"  # needs to be appended
        self.tweet = tweet
        self.embeddings_dict = {}
        self.load_glove(self.glovePath)
        self.closest_embeddings = []

    def get_glove(self, tweet):
        for word in self.tweet.split():
            newWord = word.lower()
            newWord = word.strip()
            t = self.find_closest_embeddings(self.embeddings_dict[newWord])[0:10]
            self.closest_embeddings.append((word, t))

    def load_glove(self, base_path):  # base_path is relative to current folder
        dirname = os.path.dirname(__file__)
        path = dirname + base_path

        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector

    def find_closest_embeddings(self, embedding):
        return sorted(self.embeddings_dict.keys(),
                      key=lambda sim: spatial.distance.cityblock(self.embeddings_dict[sim]
                                                                 , embedding))
# for i in embeddings_dict:
#     print(i)
#     print(find_closest_embeddings(embeddings_dict[i])[0:5])
#     print('\n')

# tsne = TSNE(n_components=2, random_state=0)
# words =  list(embeddings_dict.keys())
# vectors = [embeddings_dict[word] for word in words]
# Y = tsne.fit_transform(vectors[:1000])
# plt.scatter(Y[:, 0], Y[:, 1])
#
# for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")
# plt.show()
