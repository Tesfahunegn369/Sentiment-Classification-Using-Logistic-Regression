import numpy as np
def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or a np.array)
    Output:
        h: the sigmoid of z
    '''
    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))

    return h

def sigmoidTest():
    print("Implement sigmoid function")
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
