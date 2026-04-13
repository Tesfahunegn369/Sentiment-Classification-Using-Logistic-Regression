from utils import *

def extract_features(tweet, freqs):
    '''
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    # loop through each word in the list of words
    for word in word_l:
        # increment the word count for the positive label 1
        x[0, 1] += freqs.get((word, 1), 0)

        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0), 0)

    assert (x.shape == (1, 3))
    return x

def featureExtractionTest(train_x, freqs):
    print("# Check the feature extractor")

    # test 1
    # test on training data
    tmp1 = extract_features(train_x[0], freqs)
    print(f"Feature of {train_x[0]}: {tmp1}")

    # test 2:
    # check for when the words are not in the freqs dictionary
    tmp2 = extract_features('blorb bleeeeb bloooob', freqs)
    print(f"Feature of unseen words: {tmp2}\n")

    print("Implement feature extraction")
    if np.array_equal(tmp1, [[1.00e+00, 3.02e+03, 6.10e+01]]):
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)

    if np.array_equal(tmp2, [[1., 0., 0.]]):
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)
    print('')
