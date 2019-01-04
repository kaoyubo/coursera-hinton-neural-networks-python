import numpy as np 


def fprop(input_batch, word_embedding_weights, embed_to_hid_weights, 
          hid_to_output_weights, hid_bias, output_bias):
    """
    This method forward propogates through a neural network.
    Inputs:
        input_batch: The input data as a matrix of size numwords X batchsize 
            where, numwords is the number of words, batchsize is the number 
            of data points. So, if input_batch[i, j] = k then the ith word 
            in data point j is word index k of the vocabulary. 

        word_embedding_weights: Word embedding as a matrix of size vocab_size
            X numhid1, where vocab size is the size of the vocabulary numhid1 
            is the dimensionality of the embedding space. 
        
        embed_to_hidden_weights: Weights between the word embedding layer and 
            hidden layer as a matrix of size numhid1*numwords X numhid2, 
            numhid2 is the number of hidden units.

        hid_to_output_weights: Weights between the hidden layer and output 
            softmax unit as a matrix of size numhid2 X vocab_size.

        hid_bias: Bias of the hidden layer as a matrix of size numhid2 X 1.
        
        output_bias: Bias of the output layer as a matrix of size vocab_size X 1. 
    
    Outputs:
        embedding_layer_state: State of units in the embedding layer as a 
            matrix of size numhid1*numwords X batchsize.

        hidden_layer_state: State of the units in the hidden layer as a matrix
            of size numhid2 X batchsize. 

        output_layer_state: State of units in the ouput layer as a matrix of 
            size vocab_size X batchsize.  
    """
    
    (numwords, batchsize) = input_batch.shape 
    (vocab_size, numhid1) = word_embedding_weights.shape 
    numhid2 = embed_to_hid_weights.shape[1] 

    ## COMPUTE STATE OF WORD EMBEDDING LAYER
    # Look up the inputs word indices in the word_embedding_weights matrix
    flattened_input = np.reshape(input_batch, (1, -1), order='F')
    lookup_input_weights = word_embedding_weights[flattened_input-1, :]
    embedding_layer_state = np.reshape(lookup_input_weights, 
                                       (numhid1*numwords, -1), 
                                       order='F')
    
    ## COMPUTE STATE OF HIDDEN LAYER
    # Compute inputs to hidden units.
    repeat_hid_bias = np.repeat(hid_bias, batchsize, axis=1)
    inputs_to_hidden_units = np.dot(embed_to_hid_weights.T, embedding_layer_state) + \
                                    repeat_hid_bias 
    
    # Apply logistic activation funtion.
    # FILL IN CODE. Replace the line below by one of the options.
    hidden_layer_state = np.zeros((numhid2, batchsize))
    # Options
    # (a) hidden_layer_state = 1 / (1 + np.exp(inputs_to_hidden_units))
    # (b) hidden_layer_state = 1 / (1 - np.exp(-inputs_to_hidden_units))
    # (c) hidden_layer_state = 1 / (1 + np.exp(-inputs_to_hidden_units))
    # (d) hidden_layer_state = -1 / (1 + np.exp(-inputs_to_hidden_units))

    ## COMPUTE STATE OF OUTPUT LAYER
    # Compute inputs to softmax.
    # FILL IN CODE. Replace the line below by one of the options.
    inputs_to_softmax = np.zeros((vocab_size, batchsize))
    # Options
    # (a) inputs_to_softmax = hid

