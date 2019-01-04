import numpy as np 

# This function trains a neural network language model.
def train(epochs):
    """
    Inputs:
        epochs: Number of epochs to run.
    Output:
        model: A struct containing the learned weights and biases 
        and vocabulary.
    """


    # SET HYPERPARAMETERS HERE.
    batchsize = 100  # Mini-batch size.
    learning_rate = 0.1  # Learning rate; default = 0.1.
    momentum = 0.9  # Momentum; default = 0.9.
    numhid1 = 50  # Dimensionality of embedding space; default = 50.
    numhid2 = 200  # Number of units in hidden layer; default = 200.
    init_wt = 0.01  # Standard deviation of the normal distribution which
                    # is sampled to get the initial weights; default = 0.01

    # VARIABLES FOR TRACKING TRAINING PROGRESS.
    show_training_CE_after = 100
    show_validation_CE_after = 1000

    # LOAD DATA.
    [train_input, 
     train_target, 
     valid_input, 
     valid_target, 
     test_input, 
     test_target, 
     vocab] = load_data(batchsize)
    [numwords, batchsize, numbatches] = train_input.shape 
     vocab_size = vocab.shape[1]

    # INITIALIZE WEIGHTS AND BIASES.
    word_embedding_weights = init_wt * np.random.randn(vocab_size, numhid1)
    embed_to_hid_weights = init_wt * np.random.randn(numwords*numhid1, numhid2)
    hid_to_output_weights = init_wt * np.random.randn(numhid2, vocab_size)
    hid_bias = np.zeros((numhid2, 1))
    output_bias = np.zeros((vocab_size, 1))

    word_embedding_weights_delta = np.zeros((vocab_size, numhid1))
    word_embedding_weights_gradient = np.zeros((vocab_size, numhid1))
    embed_to_hid_weights_delta = np.zeros((numwords * numhid1, numhid2))
    hid_to_output_weights_delta = np.zeros((numhid2, vocab_size))
    hid_bias_delta = np.zeros((numhid2, 1))
    output_bias_delta = np.zeros((vocab_size, 1))
    expansion_matrix = np.identity(vocab_size)
    count = 0
    tiny = np.exp(-30)


    # TRAIN.
    for epoch in range(epochs):
        print('Epoch'.format(epoch));
        this_chunk_CE = 0;
        trainset_CE = 0;
        # LOOP OVER MINI-BATCHES.
        for m in (numbatches):
            input_batch = train_input[:, :, m]
            target_batch = train_target[:, :, m]

        # FORWARD PROPAGATE.
        # Compute the state of each layer in the network given the input batch
        # and all weights and biases
        [embedding_layer_state, 
         hidden_layer_state, 
         output_layer_state] = fprop(input_batch, 
                                     word_embedding_weights, 
                                     embed_to_hid_weights, 
                                     hid_to_output_weights, 
                                     hid_bias, output_bias)

        # COMPUTE DERIVATIVE.
        ## Expand the target to a sparse 1-of-K vector.
        flatten_input = np.reshape(target_batch, (1, -1), order='F')
        expanded_target_batch = expansion_matrix[:, flatten_input-1]
        ## Compute derivative of cross-entropy loss function.
        error_deriv = output_layer_state - expanded_target_batch

        % MEASURE LOSS FUNCTION.
        CE = -sum(sum(...
          expanded_target_batch .* log(output_layer_state + tiny))) / batchsize;
        count =  count + 1;
        this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count;
        trainset_CE = trainset_CE + (CE - trainset_CE) / m;
        fprintf(1, '\rBatch %d Train CE %.3f', m, this_chunk_CE);
        if mod(m, show_training_CE_after) == 0
          fprintf(1, '\n');
          count = 0;
          this_chunk_CE = 0;
        end
        if OctaveMode
          fflush(1);
        end


