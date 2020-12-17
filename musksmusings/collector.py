import datadotworld as dw
import json
import numpy as np
import os
import pandas as pd
import sklearn.cluster
import sklearn.decomposition
import yfinance as yf

class TweetCollector(object):
    def __init__(self):
        self.tweets = None
        self.till2017 = None
        self.from2017 = None

    def config(self, filename):
        '''
        Args:
            filename: the JSON file containing the authentication token

        Return:
            None

        Load a JSON config file at `filename` containing the authentication and 
        other information. Before running this cell, make sure that `filename` has 
        been uploaded into Colab, or is at least in the current working directory of
        this notebook. The file's contents should have the following format:

        `{"token": <YOUR_TOKEN>}`

        In this file, `<YOUR_TOKEN>` is the authentication token string for the 
        Python integration with data.world. Once the authentication token is 
        obtained, set it as an environment variable so that it can be used by the 
        datadotworld library, which is used to import datasets automatically. The 
        specific environment variable is `DW_AUTH_TOKEN`, as specified in the docs
        for `datadotworld`. 
        
        For information on obtaining this token and how it is being used in this 
        method, see the links below. You will need to create an account and generate
        your own authentication token to run this code. The first link is to the
        integration page. Once you have enabled the integration, go to the Manage
        tab and you will see you authentication token. The second link is to help
        understand how the library is configured and how datasets are imported.

        https://data.world/integrations/python 
        https://help.data.world/hc/en-us/articles/360039429733-Python-SDK
        '''
        with open(filename) as f:
            config = json.load(f)
            token = config['auth_token']
            os.environ['DW_AUTH_TOKEN'] = token

    def get_tweets(self):
        '''
        A function that uses the `datadotworld` library, to load the two datasets
        used in this project, and the dataframes in this datasets are extracted.
        Below are links to the datasets being used. 
        
        https://data.world/adamhelsinger/elon-musk-tweets-until-4-6-17 (2010-2017)
        https://data.world/barbaramaseda/elon-musk-tweets (2017 onwards)

        The object returned from `dw.load_dataset()` contains a dataframe that needs
        to be extracted from a dictionary of other dataframes. Because there is only
        one dataframe in each dataset, it is extracted using set-list conversion, 
        avoiding the need to use a key to look it up.
        '''
        till2017data = dw.load_dataset('adamhelsinger/elon-musk-tweets-until-4-6-17')
        from2017data = dw.load_dataset('barbaramaseda/elon-musk-tweets')
        self.till2017 = list(till2017data.dataframes.values())[0]
        self.from2017 = list(from2017data.dataframes.values())[0]        

    def clean_tweets(self):
        '''
        Return:
            tweets: the cleaned dataframe containing the Tweets and other information (no stocks)
        
        Clean the dataframes obtained from `get_dataframes()`. Those dataframes 
        should be stored somewhere before calling this function in case changes need
        to be reverted. The cleaning process is detailed below. 
        '''
        # Make a copy of the original, remove binary string identifiers, rename columns, set index
        till2017 = self.till2017.copy()
        till2017['text'] = till2017['text'].str.strip('b\'\"')
        till2017 = till2017.rename(columns={'created_at': 'timestamp'})
        till2017 = till2017.reindex(columns=['id', 'timestamp', 'text'])
        till2017 = till2017.set_index('id')

        # Make a copy of the original, extract and format ids, drop and rename some columns, set index
        from2017 = self.from2017.copy()
        from2017['linktotweet'] = from2017['linktotweet'].str.strip('http://twitter.com/elonmusk/status/')
        from2017['linktotweet'] = from2017['linktotweet'].astype(int)
        from2017 = from2017.drop(labels=['username', 'tweetembedcode'], axis=1)
        from2017 = from2017.rename(columns={'createdat': 'timestamp', 'linktotweet': 'id'})
        from2017 = from2017.reindex(columns=['id', 'timestamp', 'text'])
        from2017 = from2017.set_index('id')

        # Concatenate dataframes, drop duplicates
        tweets = pd.concat([till2017, from2017])
        tweets = tweets.drop_duplicates()

        # Separate timestamp into date and time, so that stock data can be found
        tweets['date'] = pd.to_datetime(tweets['timestamp']).dt.date
        tweets['time'] = pd.to_datetime(tweets['timestamp']).dt.time

        # Reset index, remove id and timestamp columns, index by date column
        tweets = tweets.reset_index()
        tweets = tweets.drop(labels=['id', 'timestamp'], axis=1)
        tweets = tweets.reindex(columns=['date', 'time', 'text'])
        tweets = tweets.set_index('date')
        tweets = tweets.sort_index()

        # Label retweets, replies, and regular tweets
        tweets['label'] = 'tweet'
        tweets.loc[tweets.text.str.startswith("@", na=False), 'label'] = 'reply'
        tweets.loc[tweets.text.str.startswith('RT', na=False), 'label'] = 'rt'
        
        # Export tweets to CSV
        tweets.to_csv('tweets.csv')
        self.tweets = tweets
        
class StockCollector(object):
    def __init__(self):
        self.stocks = None
        self.tsla = None
        self.nasdaq = None
        self.sp500 = None

    def get_stocks(self):        
        '''
        A function that uses the yfinance package to pull historical stock data
        from Yahoo Finance. The stocks of interest are TSLA, NASDAQ, and SP500.
        '''
        self.tsla = yf.Ticker('TSLA').history(period='max')
        self.nasdaq = yf.Ticker('^IXIC').history(period='max')
        self.sp500 = yf.Ticker('^GSPC').history(period='max')

    def clean_stocks(self):
        '''
        A function that cleans and combines historical stock data from yfinance
        and exports it as a CSV to be used in the model.
        '''
        def clean_yfdata(yfdata):
            '''
            A function for cleaning yfinance historical stock data. It renames
            and removes some columns, normalizes the prices so that it can be
            compared to the market indexes, and calculates daily price deltas
            '''
            yfdata = yfdata.copy()
            yfdata = yfdata.reset_index()
            yfdata = yfdata.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close'})
            yfdata = yfdata.drop(labels=['Dividends', 'Stock Splits', 'Volume'], axis=1)
            yfdata = yfdata.set_index('date')
            yfdata = yfdata.sort_index()
            yfdata = (yfdata - yfdata.min()) / (yfdata.max() - yfdata.min())
            yfdata['range'] = data.high - data.low
            yfdata['change'] = data.close - data.open
            yfdata = yfdata.fillna(0)
            return yfdata

        tsla = clean_yfdata(self.tsla)
        nasdaq = clean_yfdata(self.nasdaq[self.nasdaq.index.isin(tsla.index.array)])
        sp500 = clean_yfdata(self.sp500[self.sp500.index.isin(tsla.index.array)])
        stocks = tsla - nasdaq
        
        feats = 2
        pca = sklearn.decomposition.PCA(feats)
        pca.fit(stocks.to_numpy())
        reduced = pca.transform(stocks.to_numpy())
        
        clusters = 3
        kmeans = sklearn.cluster.KMeans(clusters)
        classification = kmeans.fit_predict(reduced)
        stocks['action'] = classification
        
        stocks.to_csv('stocks.csv')
        self.stocks = stocks
