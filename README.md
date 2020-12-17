This repository is a clone from the GitHub enterprise server for Georgia Tech. This project was made for CS 4641 - Machine Learning, a group project based course at Georgia Tech. The members of the group (in alphabetical order) are: Abhirup Ghosh, Aniruddha Murali, Rahul Katre, Srishti Mendiratta. 

# Introduction/Background

Elon Musk, the founder and CEO of Tesla, is known for his somewhat erratic behavior. His statements have a profound effect on the stock prices of his company and its competitors. Most of his statements come as a result of his tweets on Twitter, totaling ~7000 tweets since 2012. We want to find how each tweet impacts trading strategy for TSLA stock.

# Problem definition

We devise a trading strategy consisting of buying, selling, or holding in a single day, based on any the contents of tweets Musk has made that day. Given a tweet, we classify the tweet as something that will change the stock to a 'buy' rating i.e. predict the stock price goes up, 'sell' rating i.e. predict the stock price goes down, or 'hold', if the tweet doesn't impact the stock value.

# Methods

## Data Collection

To collect tweets made by Musk, we use two datasets from [data.world], linked below. The first source is an archive of Musk's tweets until a date in 2017, and the second source is a daily-updating collection of Musk's tweets, starting from 2017. The advantage of the second dataset is that it updates daily, so we do not need to work with any Python API to get the latest tweets ourselves. 

In order to collect data about stocks, we use the yfinance Python library to get a historical time series of TSLA's daily open, high, low, and close prices from Yahoo Financ. We also obtain a time series for the NASDAQ index from Yahoo Finance in order to normalize TSLA's value relative to the market. 

## Data Cleaning

In order to clean the tweet data, we merge the separate tweet datasets. Using [data.world]'s Python library, we extract Pandas DataFrames from the datasets. The first DataFrame (till 2017) contains columns for the tweet ID, the timestamp, and the tweet text. The text contains binary string formatting artifacts that need to be removed, and the timestamp column needs to be formatted as a DateTime column. 

The second dataframe contains columns for the tweet text, the username, the link to the tweet, the HTML embed code for the tweet, and the timestamp. The username and embed code columns are dropped, and the timestamp column is formatted as a DateTime column. The link column is useful because the link to each tweet, the tweet ID can be found at the end of the URL, so while cleaning this column, all preceding characters are removed so that the ID for each tweet can be obtained.

To merge the two DataFrames, we first ordered the columns in each DataFrame identically (ID, timestamp, tweet text). We then assigned the ID column as the index for each DataFrame. Because the two datasets originally had some overlap in tweets, we found that the easiest way to remove duplicate tweets would be by the ID of the tweet. When merging the two DataFrames, we dropped all duplicate rows. Upon merging, we removed the ID column and made the timestamp column the new indexing column so that the stock time-series data could be mapped to it. 

In order to process the text for each tweet, we label each tweet as a regular tweet, a reply to another tweet, or a retweet. A pattern that helps us identify the category was that all replies start with a "@" character to mention the user Musk was replying to. Retweets are similar, as they contain the prefix "RT". By scanning the starting characters of each tweet, we are able to categorize each tweet. 

The time-series data for TSLA and NASDAQ was obtained using the yfinance Python library, which obtains data from Yahoo Finance based on an input ticker. Using this library, we obtained the historical data, which contained daily data points for the stock's open, high, low, and closing prices, organizes by a timestamp column. There were other columns for volume and splits, but these columns were dropped. We first normalize TSLA relative to the market by subtracting the NASDAQ values from the same day. To score the performance of the stock each day, we calculate the range and change in daily values. Range is defined by the difference between a daily high and daily low, while change is defined by the difference between daily close and daily open.

The stock dataframe is merged with the tweets dataframe, and resulting dataframe now contains every single tweet made by Musk without duplicates, and the associated TSLA stock values for the day that tweet was made. The key metric is the change in closing stock price, which we used to identify influential tweets. All cleaned DataFrames were exported as comma-separated-value files to be used in our modeling algorithms. 

## Unsupervised Learning For Data Labeling

In order for later steps to function correctly, we need to label each day of the market normalized TSLA values as a buy, sell, or hold day. To prepare the data, we reduce the size of the feature space of the stock data. Initially, the stock data has features for open, close, high, low, range, and change values. Using principal component analysis (PCA), we reduce the amount of features from 6 to 2. We choose this amount of features because it retains over 99% of the variance. [1]

With the feature space reduced, the data is ready to be classified using an unsupervised learning implementation. Because we want to label the data without already having any labels, a clustering algorithm works best. We choose the KMeanse algorithm because we want a hard classifier and exactly 3 clusters. The GMM algorithm is a soft classifier, and the DBSCAN algorithm does not result in exactly 3 clusters, which is why we choose KMeans. After running the algorithm, we add the resulting labels as another column in the stocks dataframe before merging it with the tweets dataframe. 

![Market normalized close](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/stocksclose.png) ![TSLA close](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/tslaclose.png)

The results indicate a good classification. On the left is the market normalized closing prices that are reduced before the KMeans algorithm is applied. Using the results of the algorithm, the labels are applied to the TSLA closing prices. From this, we reason that a purple is a sell, yellow is a hold, and green is a buy, as it indicates that a trader should buy the share when it is significantly underperforming the rest of the market and not just because it is at a low point in the share history till that date. 

## Supervised Learning Model

We made use of three major supervised learning models: Ensemble Learning, Naive Bayes, and a Neural Network. In order to get desirable results from these methods, we had to process our input using the following methodologies:
- TFIDF (Term Frequency — Inverse Document Frequency) which is a technique to quantify the relevancy of a word in a document. If a word occurs multiple times in a document, we should boost its relevance, whereas if a word occurs many times in a document but also along many other documents, the word likely is just frequent and doesn’t have much relevance. In order to implement this, we preprocess tweets, create a TFIDF matrix from set of tweets and remove features that have TFIDF of 0 in all tweets.
- In addition, we resorted to balancing our dataset, as we had 852 tweets labeled as sell, 65587 tweets labelled as hold, and 2153 tweets labelled as buy. This is very unbalanced, and in several cases, led to poor accuracies in the minority classes i.e. buy/sell. We upsampled the data, instead of downsampling (which would have led to a very few number of overall tweets). Finally, we had 5587 tweets in every class, with a random set of tweets being repeated in the buy/sell categories.

### Ensemble Learning
Ensemble Models use results from dimensionality reduction (PCA) to determine classes to predict the possible classes: buy, sell, or keep on hold. Models are trained with the TFIDF matrix created from the set of collected tweets, and accuracy was calculated with F1 score to deal with imbalanced classes. Our parameters and accuracy for various techniques are listed below:
- Random Forest (Hyperparameterization: max_depth=12, n_estimators=10): Accuracy: 66.4%
- Gradient Boosting (Hyperparameterization: max_depth=5): Accuracy: 73.8%

### Naive Bayes
Our results using a Naive Bayes classifier are below:

*Prior to upsampling*

![Accuracy details](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/nb_prior_accuracy.png)
![Confusion matrix](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/confusion_nb_prior.png)

*After upsampling*

![Accuracy details](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/nb_post_accuracy.png)
![Confusion matrix](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/confusion_nb_post.png)

### Neural Network
We used a neural network with 3 hidden layers, each with 12, 8, 8 neurons respectively. As we reduced the number of features from our TFIDF matrix to 4000, we have 4000 input neurons, and 3 output neurons (one for each class - buy, hold, sell). Similar to the Naive Bayes model, upsampling led to much better results in our case. Below, you will find the model accuracy, model loss, and confusion matrices for before and after upsampling the data.

*Prior to upsampling*

![Model Accuracy](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/NN-prior-acc.png)
![Model Loss](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/NN-prior-loss.png)
![Confusion matrix](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/NN-prior-confusion.png)

*After upsampling*

![Model Accuracy](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/NN-post-acc.png)
![Model Loss](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/NN-post-loss.png)
![Confusion matrix](https://github.gatech.edu/aghosh74/Musks-Musings/blob/master/images/NN-post-confusion.png)

A few notes about our Neural Network:
- As we started off with a very low number of tweets for a neural network, which often requires >100,000 datapoints to lead to a good accuracy, it was obvious that we wouldn't have great results with only ~7500 tweets, which were approximately ~15,000 after upsampling.
- We used an Adam classifier, which has been said to not perform too well with smaller datasets. Perhaps, using SGD to update our weights may have performed better.
- Finally, using a Hingle loss function is said to perform better with smaller datasets. Although we didn't explore this in our project, it might have led to better results.

# Results
Individual results for different types of models are discussed under the method descriptions itself. NOTE: All accuracy figures are based off a 70-30 training-test dataset split.

# Discussion

### Difficulties
 - Very few overall tweets were available to use. Overall, a better alternative to make a tweet-stock relation would be to collate the top CEOs in the industry and relate their tweets to stock price changes in their respective companies. This would expand the pool of available tweets, thereby allowing Neural Networks to perform better.
 - The dataset in general was very unbalanced. While upsampling reduces the impact of an unbalanced dataset to a certain degree, it doesn't completeley eliminate the weaknesses of unbalanced datasets. If we are able to collect a larger pool of tweets, downsampling would lead to fewer "fake" tweets and thereby lead to a more accuracte model.
 
## Potential improvements
- Using GlOvE to mask the input tweet to be assessed would allow us to stay within vocabulary of our TFIDF matrix.
- GlOvE and Word2vec don’t know how to learn the representation of out-of-vocabulary words. GlOvE takes up a lot of storage, and every time you change hyper-parameters that are related to the co-occurrence matrices that GlOvE uses, you have to reconstruct the matrices.
- At times, the 'sequence' of tweets leads to a greater impact on the stock price. In order to take into account a series of tweets, we may use LSTMs. [3]

# Conclusion
Overall, our Neural Network performed the best with an accuracy of ~85% (however, with a higher loss given we ran the NN for ~150 epochs). Given the overall outcomes and all factors considered, our Multinomial Naive Bayes model gave us the second best accuracy of ~75%.

### Data Sources

- Tweets by Elon Musk till 2017: https://data.world/adamhelsinger/elon-musk-tweets-until-4-6-17
- Tweets by Elon Musk from 2017: https://data.world/barbaramaseda/elon-musk-tweets
- Historical Yahoo Finance TSLA stock data: https://finance.yahoo.com/quote/TSLA/history 
- Historical Yahoo Finance NASDAQ index data: https://finance.yahoo.com/quote/%5EIXIC/history

### References
  [1]: Zahra Berradia, Mohamed Lazaara. (2019) Integration of Principal Component Analysis and Recurrent Neural Network to Forecast the Stock Price of Casablanca Stock Exchange
  
  [2]: Pennington, C. (2014). GloVe: Global Vectors for Word Representation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532–1543). Association for Computational Linguistics.
  
  [3]: Moghar, A., & Hamiche, M. (2020). Stock Market Prediction Using LSTM Recurrent Neural Network. *Procedia Computer Science,* *170*, 1168-1173. doi:10.1016/j.procs.2020.03.049

  
