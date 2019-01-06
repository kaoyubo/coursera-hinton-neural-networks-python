import numpy as np 
from scipy.io import loadmat 


def load_data(N):
    """
    This method loads the training, validation and test set. 
    It also devides the training into mini-batches.
    Inputs:
        N: Mini-batch size.
    Outputs:
        train_input: An array of size D X N X M, where 
            D: number of input dimensions (in this case, 3).
            N: size of each mini-batch (in this case, 100).
            M: number of minibatches.
        train_target: An array of size 1 X N X M
        valid_input: An array of size D X number of points in the validation set. 
        test: An array of size D X number of points in the test set.
        vocab: Vocabulary containing indes to word mapping.
    """
    
    data = loadmat('C:/Users/kaoa/Downloads/assignment2/assignment2/data.mat')
    numdims = data['data']['trainData'][0][0].shape[0]
    D = numdims - 1
    M = int(np.floor(data['data']['trainData'][0][0].shape[1]/N))

    train_data = data['data']['trainData'][0][0]
    train_input = np.reshape(train_data[:D, :N*M], (D, N, M), order='F')
    train_target = np.reshape(train_data[D, :N*M], (1, N, M), order='F')

    valid_data = data['data']['validData'][0][0]
    valid_input = valid_data[:D, :]
    valid_target = valid_data[D, :]

    test_data = data['data']['testData'][0][0]
    test_input = test_data[:D, :]
    test_target = test_data[D, :]

    vocab = data['data']['vocab'][0][0][0]

    return (train_input, 
            train_target, 
            valid_input, 
            valid_target, 
            test_input, 
            test_target, 
            vocab)

