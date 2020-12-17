# An example of getting the Twitter and market data, cleaning it, and combining it. 

import os
import json
import datadotworld as dw
import numpy as np
import pandas as pd
import yfinance as yf

import musksmusings as mm

filename = 'config.json'
tc = mm.TweetCollector()
tc.config(filename)
tc.get_tweets()
tc.clean_tweets()
tweets = tc.tweets

sc = mm.StockCollector()
sc.get_stocks()
sc.clean_stocks()
stocks = sc.stocks

tweetstocks = tweets.join(stocks, how='inner')
tweetstocks = tweetstocks.sort_values('delta', ascending=True)
tweetstocks.to_csv('tweetstocks.csv')
