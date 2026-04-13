import nltk
from nltk.corpus import twitter_samples
import numpy as np

def load_tweets(path):
    # This creates 'data' folder and downloads the twitter sample corpora to that folder.
    nltk.download('twitter_samples', download_dir=path)
    nltk.download('stopwords', download_dir=path)
    nltk.data.path.append(path)

    # select the set of positive and negative tweets
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')

    # split the data into two pieces, one for training and one for testing (validation set)
    test_pos = all_positive_tweets[4000:]
    train_pos = all_positive_tweets[:4000]
    test_neg = all_negative_tweets[4000:]
    train_neg = all_negative_tweets[:4000]

    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

    # combine positive and negative labels
    train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
    test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

    # Print the shape train and test sets
    print("train_y.shape = " + str(train_y.shape))
    print("test_y.shape = " + str(test_y.shape))

    return train_x, train_y, test_x, test_y
