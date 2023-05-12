from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.regularizers import l2

from shorttext.utils.classification_exceptions import UnequalArrayLengthsException


def dense_word_embed(nb_labels, dense_nb_nodes=[], dense_actfcn=[], vecsize=300, reg_coef=0.1, final_activiation='softmax', optimizer='adam'):
    """Return layers of a dense neural network.

    :param nb_labels: number of class labels
    :param dense_nb_nodes: number of nodes in each layer (default: [])
    :param dense_actfcn: activation functions for each layer (default: [])
    :param vecsize: length of the embedded vectors in the model (default: 300)
    :param reg_coef: regularization coefficient (default: 0.1)
    :param final_activiation: activation function of the final layer (default: softmax)
    :param optimizer: optimizer for gradient descent, options: sgd, rmsprop, adagrad, adadelta, adam, adamax, or nadam (default: adam)
    :return: keras sequential model for dense neural network
    """
    if len(dense_nb_nodes) != len(dense_actfcn):
        raise UnequalArrayLengthsException(dense_nb_nodes, dense_actfcn)

    model = Sequential()

    if len(dense_nb_nodes) == 0:
        model.add(Dense(nb_labels, input_shape=(vecsize,), kernel_regularizer=l2(reg_coef)))
    else:
        model.add(Dense(dense_nb_nodes[0], input_shape=(vecsize,), activation=dense_actfcn[0], kernel_regularizer=l2(reg_coef)))

        for idx, nb_nodes in enumerate(dense_nb_nodes[1:], 1):
            model.add(Dense(nb_nodes, activation=dense_actfcn[idx], kernel_regularizer=l2(reg_coef)))

        model.add(Dense(nb_labels, kernel_regularizer=l2(reg_coef)))

    model.add(Activation(final_activiation))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model