from typing import List
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.regularizers import l2
from shorttext.utils.classification_exceptions import UnequalArrayLengthsException


def build_dense_word_embed_model(
    nb_labels: int,
    vecsize: int = 300,
    dense_nb_nodes: List[int] = [],
    dense_activation_functions: List[str] = [],
    reg_coef: float = 0.1,
    final_activation_function: str = 'softmax',
    optimizer: str = 'adam'
) -> Sequential:
    """
    Builds a dense neural network model for word embeddings.

    :param nb_labels: The number of class labels.
    :param vecsize: The length of the embedded vectors in the model. (Default: 300)
    :param dense_nb_nodes: List of integers specifying the number of nodes in each layer. (Default: [])
    :param dense_activation_functions: List of activation functions for each layer. (Default: [])
    :param reg_coef: Regularization coefficient. (Default: 0.1)
    :param final_activation_function: Activation function of the final layer. (Default: softmax)
    :param optimizer: The optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: adam)
    :return: A keras sequential model for dense neural network.
    """
    if len(dense_nb_nodes) != len(dense_activation_functions):
        raise UnequalArrayLengthsException(dense_nb_nodes, dense_activation_functions)

    nb_layers = len(dense_nb_nodes)
    model = Sequential()

    # Add input layer
    if nb_layers == 0:
        model.add(Dense(nb_labels, input_shape=(vecsize,), kernel_regularizer=l2(reg_coef)))
    else:
        model.add(Dense(dense_nb_nodes[0], input_shape=(vecsize,), activation=dense_activation_functions[0], kernel_regularizer=l2(reg_coef)))

        # Add hidden layers
        for nb_nodes, activation in zip(dense_nb_nodes[1:], dense_activation_functions[1:]):
            model.add(Dense(nb_nodes, activation=activation, kernel_regularizer=l2(reg_coef)))

        # Add output layer
        model.add(Dense(nb_labels, kernel_regularizer=l2(reg_coef)))

    # Add final activation layer
    model.add(Activation(final_activation_function))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model