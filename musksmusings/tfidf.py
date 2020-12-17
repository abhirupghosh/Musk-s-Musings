import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re

class TFIDF:
    # Defining a Function to clean up the reviews
    def text_preprocess(ds: pd.Series) -> pd.Series:
        for m in range(len(ds)):
            main_words = re.sub('[^a-zA-Z]', ' ', ds[m])  # Retain only alphabets
            main_words = (main_words.lower()).split()
            main_words = [w for w in main_words if not w in set(stopwords.words('english'))]  # Remove stopwords

            lem = WordNetLemmatizer()
            main_words = [lem.lemmatize(w) for w in main_words if len(w) > 1]  # Group different forms of the same word

            main_words = ' '.join(main_words)
            ds[m] = main_words
        return ds


    def get_tfidf(new_tweet):
        df = pd.read_csv('tweetstocks.csv')
        df = resample(df)
        tweets = df['text'].values.astype('U')
        tweets = np.append(tweets, new_tweet)

        tweets = text_preprocess(tweets)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=4000)

        X = vectorizer.fit_transform(tweets)
        tfidf_matrix = X.todense()  # sparse matrix from TFIDF
        return remove_zero_tf_idf(tfidf_matrix)

    def resample(df):
        df_majority = df[df.action == 1]
        df_med = df[df.action == 2]
        df_minority = df[df.action == 0]

        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=5587,  # to match majority class
                                         random_state=123)

        df_med_upsampled = resample(df_med,
                                    replace=True,  # sample with replacement
                                    n_samples=5587,  # to match majority class
                                    random_state=123)

        df_upsampled = pd.concat([df_majority, df_minority_upsampled, df_med_upsampled])
        return df_upsampled

    def remove_zero_tf_idf(Xtr, min_tfidf=0.04):
        D = Xtr  # .toarray() # convert to dense if you want
        D[D < min_tfidf] = 0
        tfidf_means = np.mean(D, axis=0)  # find features that are 0 in all documents
        D = np.delete(D, np.where(tfidf_means == 0)[0], axis=1)  # delete them from the matrix
        return D


    def get_label():
        df = pd.read_csv('tweetstocks.csv')
        labels = df['action'].values.astype('U')
        return labels