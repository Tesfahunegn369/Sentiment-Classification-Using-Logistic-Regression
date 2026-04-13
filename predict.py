from feature_extraction import *
from sigmoid import *

# Q4: Implement prediction of sentiments using tweets, with the functions you implemented (20 pts)
def predict_tweet(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))

    ### END CODE HERE ###

    return y_pred

def predictionTest(freqs, theta):
    print("# Check the prediction")

    sentiments = []
    for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
        pred = predict_tweet(tweet, freqs, theta)
        print('%s -> %f' % (tweet, pred))
        sentiments.append(round(float(pred), 6))

    # Feel free to check the sentiment of your own tweet below
    my_tweet = 'I am learning :)'
    predict_tweet(my_tweet, freqs, theta)
    print('\n%s -> %f\n' % (my_tweet, predict_tweet(tweet, freqs, theta)))

    print("Q4: Implement prediction of sentiments using tweets (20 pts)")
    if np.sum((np.array(sentiments) - np.array([0.518562, 0.494329, 0.515312, 0.515449, 0.530868, 0.546228, 0.561501])) ** 2) < 1e-4:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)
    print('')



