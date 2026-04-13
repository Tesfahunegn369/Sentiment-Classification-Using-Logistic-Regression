from load_data import load_tweets
from optimization import *
from evaluate import *

# Load nltk tweeter sample corpora
path = "./data"
train_x, train_y, test_x, test_y = load_tweets(path)

# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# check the output
print(f"type of freqs = {type(freqs)}")
print(f"length of freqs = {len(freqs.keys())}\n")

# test the function below
print(f'# This is an example of a positive tweet: \n{train_x[0]}\n')
print(f'# This is an example of the processed version of the tweet: \n{process_tweet(train_x[0])}\n')

sigmoidTest()  # Q1
costFuncTest()
gradientDescentTest()  # Q2
featureExtractionTest(train_x, freqs)  # Q3

## Sentiment classification using nltk tweeter sample dataset
# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}\n")

predictionTest(freqs, theta)  # Q4
evalTest(test_x, test_y, freqs, theta)  # Q5

print("Congratulations! You have completed all tasks.\n")

## What data did the model predict incorrectly?
print('Error Analysis')
for x,y in zip(test_x,test_y):
    y_hat = predict_tweet(x, freqs, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', process_tweet(x))
        print('%d\t%0.8f\t%s\n' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))

## Feel free to change the tweet below
my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'
print(my_tweet)
print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else:
    print('Negative sentiment')