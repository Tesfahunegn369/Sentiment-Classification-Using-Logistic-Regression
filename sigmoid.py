import numpy as np

# Q1: Implement sigmoid function (10 pts)
def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or a np.array)
    Output:
        h: the sigmoid of z
    '''

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###

    return h

def sigmoidTest():
    print("Q1: Implement sigmoid function (10 pts)")
    result_0 = sigmoid(0)
    result_4_92 = sigmoid(4.92)

    #print("sigmoid(0):", result_0)  
    #print("sigmoid(4.92):", result_4_92)  

    if (result_0 - 0.50) ** 2 < 1e-10:
        print('SUCCESS')
    else:
        print('FAIL for sigmoid(0)')

    if (result_4_92 - 0.9927537604041685) ** 2 < 1e-10:
        print('SUCCESS')
    else:
        print('FAIL for sigmoid(4.92)')

    print('')