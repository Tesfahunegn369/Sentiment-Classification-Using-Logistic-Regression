from sigmoid import *
def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    # get 'm', the number of rows in matrix x
    m = len(y)

    print("Optimization starts")
    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)
        
        # get the sigmoid of z
        h = 1 / (1 + np.exp(-z))

        # calculate the cost function
        J = - (1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

        # check the loss value at every 100 epochs
        if i % 100 == 0:
            print(f"Loss: {J}")

        # update the weights theta
        theta = theta - (alpha / m) * np.dot(x.T, (h - y))
        
    J = float(J)
    return J, theta

def costFuncTest():
    # verify that when the model predicts close to 1, but the actual label is 0, the log loss is a large positive value
    print("# Loss when the model predicts close to 1, but the actual label is 0: ")
    print(f"Loss: {-1 * (1 - 0) * np.log(1 - 0.9999)}") # loss is about 9.2

    # verify that when the model predicts close to 0 but the actual label is 1, the log loss is a large positive value
    print("# Loss when the model predicts close to 0, but the actual label is 1: ")
    print(f"Loss: {-1 * np.log(0.0001)}\n") # loss is about 9.2

def gradientDescentTest():
    print("# Check the gradient descent function")

    # Construct a synthetic test case using numpy PRNG functions
    np.random.seed(1)
    # X input is 10 x 3 with ones for the bias terms
    tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
    # Y Labels are 10 x 1
    tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

    # Apply gradient descent
    tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
    print(f"The cost after training is {tmp_J:.8f}.")
    print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}\n")

    print("Implement gradient descent")
    if tmp_J < 0.675:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)

    if np.sum((np.squeeze(tmp_theta) - [4.107e-07, 3.566e-04, 7.309e-05]) ** 2) < 1e-10:
        print('SUCCESS')
    else:
        print('FAIL')
        exit(1)
    print('')
