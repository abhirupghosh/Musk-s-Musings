from tfidf import TFIDF

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import Sequential
from keras.layers import Dense
import pandas
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, new_tweet):
        X = TFIDF.get_tfidf(new_tweet)
        self.new_tweet = X[-1]
        self.X = X[:-1]
        self.y = TFIDF.get_label()
        self.encode()
        self.build_model()
        self.y_outs = self.model.predict(self.new_tweet)

    def get_class(self):
        return self.y_outs

    def encode(self):
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                            random_state = 42)
        '''
        encoder = LabelEncoder()
        encoder.fit(self.y)
        encoded_y = encoder.transform(self.y)
        # convert integers to dummy variables (i.e. one hot encoded)
        self.encoded_y = np_utils.to_categorical(encoded_y)
        '''
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_y_train = encoder.transform(y_train)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y_train = np_utils.to_categorical(encoded_y_train)
    
        encoder = LabelEncoder()
        encoder.fit(y_test)
        encoded_y_test = encoder.transform(y_test)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y_test = np_utils.to_categorical(encoded_y_test)
        '''
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(12, input_dim=4000, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(3, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.X, self.encoded_y, epochs=150, batch_size=10)

