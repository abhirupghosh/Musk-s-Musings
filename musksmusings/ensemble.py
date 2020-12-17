from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd

from tfidf import TFIDF


'''
def process_data():
    df = pd.read_csv('tweetstocks.csv')
    X = get_tfidf()
    y = df['action']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

def random_forest():
    X_train, X_test, y_train, y_test = process_data()
    clf = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = f1_score(y_test, pred, average='micro')
    print("Score="+ str(score))
    return score

def gradient_boost():
    X_train, X_test, y_train, y_test = process_data()
    clf = GradientBoostingClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = f1_score(y_test, pred, average='micro')
    print("Score="+ str(score))
    return score
'''


class Ensemble:
    def __init__(self, tweet):
        X = TFIDF.get_tfidf(tweet)
        self.new_tweet = X[-1]
        self.X = X[:-1]
        self.y = TFIDF.get_label()
        self.gradient_boost()

    def random_forest(self):
        # F1 score around 64-68% 
        clf = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=0)
        clf.fit(self.X, self.y)
        self.pred = clf.predict(self.X)
        print(self.pred)

    def gradient_boost(self):
        # F1 score around 72-73% 
        clf = GradientBoostingClassifier(max_depth=5, random_state=0)
        clf.fit(self.X, self.y)
        self.pred = clf.predict(self.X)
        print(self.pred)
    
    def get_class(self):
        return self.pred