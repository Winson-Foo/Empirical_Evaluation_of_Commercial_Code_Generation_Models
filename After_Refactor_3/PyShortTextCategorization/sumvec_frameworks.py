from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.regularizers import l2

from shorttext.utils.classification_exceptions import UnequalArrayLengthsException


def create_dense_layers(dense_nb_nodes, dense_actfcn, vecsize, reg_coef):
    """ Create dense layers for neural network.

    :param dense_nb_nodes: number of nodes in each layer
    :param dense_actfcn: activation function for each layer
    :param vecsize: length of the input vector
    :param reg_coef: regularization coefficient
    :return: list of dense layers
    """
    layers = []
    for i in range(len(dense_nb_nodes)):
        if i == 0:
            layer = Dense(dense_nb_nodes[i], input_shape=(vecsize,), activation=dense_actfcn[i],
                          kernel_regularizer=l2(reg_coef))
        else:
            layer = Dense(dense_nb_nodes[i], activation=dense_actfcn[i], kernel_regularizer=l2(reg_coef))
        layers.append(layer)
    return layers


def DenseWordEmbed(nb_labels, dense_nb_nodes=None, dense_actfcn=None, vecsize=300, reg_coef=0.1,
                   final_activation='softmax', optimizer='adam'):
    """ Return layers of dense neural network.

    :param nb_labels: number of class labels
    :param dense_nb_nodes: list of number of nodes in each layer (default: None)
    :param dense_actfcn: list of activation functions for each layer (default: None)
    :param vecsize: length of the embedded vectors in the model (default: 300)
    :param reg_coef: regularization coefficient (default: 0.1)
    :param final_activation: activation function of the final layer (default: softmax)
    :param optimizer: optimizer for gradient descent (default: adam)
    :return: keras sequential model for dense neural network
    """
    if dense_nb_nodes is None:
        dense_nb_nodes = []
    if dense_actfcn is None:
        dense_actfcn = []

    # Check if the length of dense_nb_nodes and dense_actfcn are equal
    if len(dense_nb_nodes) != len(dense_actfcn):
        raise UnequalArrayLengthsException(dense_nb_nodes, dense_actfcn)

    # Create dense layers
    layers = create_dense_layers(dense_nb_nodes, dense_actfcn, vecsize, reg_coef)

    # Add output layer
    layers.append(Dense(nb_labels, kernel_regularizer=l2(reg_coef)))

    # Add final activation layer
    layers.append(Activation(final_activation))

    # Create model
    model = Sequential(layers)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model