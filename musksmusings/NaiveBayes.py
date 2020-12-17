tweet = "A la guerre comme a la guerre"
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tfidf import TFIDF
from sklearn.naive_bayes import MultinomialNB
import re

class NaiveBayes:
    def __init__(self, tweet):
        new_tweet = tweet
        X = TFIDF.get_tfidf(new_tweet)
        self.new_tweet = X[-1]
        self.X = X[:-1]
        self.y = TFIDF.get_label()
        self.nb()

    def nb(self):
        classifier = MultinomialNB()
        classifier.fit(self.X, self.y)

        # y_pred = classifier.predict(X_test)
        self.y_newpred = classifier.predict(self.new_tweet)
        print(self.y_newpred)

    def get_class(self):
        return self.y_newpred
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.001,
                                                    random_state = 0)
'''


'''
# Classification metrics
from sklearn.metrics import accuracy_score, classification_report
classification_report = classification_report(y_test, y_pred)


Accuracy:  0.6697674418604651 if 7000 features
Accuracy:  0.6546511627906977 if no limit on features
Accuracy:  0.6744186046511628 if 4000 features

print('\n Accuracy: ', accuracy_score(y_test, y_pred))
print('\nClassification Report')
print('======================================================')
print('\n', classification_report)
'''