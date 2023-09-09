import json

import numpy as np
import gensim

from .s2skeras import Seq2SeqWithKeras, loadSeq2SeqWithKeras, kerasseq2seq_suffices
from ..charbase.char2vec import SentenceToCharVecEncoder
from shorttext.utils import compactmodel_io as cio

# Define suffixes for saving and loading the model
seq2seq_suffices = kerasseq2seq_suffices + ['_dictionary.dict', '_seq2seq.json']


class CharBasedSeq2SeqGenerator(cio.CompactIOMachine):
    """Class implementing a character-based sequence-to-sequence (seq2seq) learning model.

    This class implements the seq2seq model at the character level. This class calls
    :class:`Seq2SeqWithKeras`.

    Reference:

    Oriol Vinyals, Quoc Le, "A Neural Conversational Model," arXiv:1506.05869 (2015). [`arXiv
    <https://arxiv.org/abs/1506.05869>`_]
    """

    def __init__(self, sentence_to_char_vector_encoder, latent_dimension, max_sentence_length):
        """Instantiate the class.

        :param sentence_to_char_vector_encoder: an instance of the SentenceToCharVecEncoder class
        :param latent_dimension: number of latent dimension
        :param max_sentence_length: maximum length of a sentence
        """

        # Call the constructor of the parent class.
        cio.CompactIOMachine.__init__(self, {'classifier': 'seq2seq'}, 'seq2seq', seq2seq_suffices)

        # If no one-hot encoder passed in, no compilation will be performed.
        self.compiled = False

        # Store the remaining parameters to be used in training and inference.
        if sentence_to_char_vector_encoder is not None:
            self.encoder = sentence_to_char_vector_encoder
            self.dictionary = self.encoder.dictionary
            self.n_elements = len(self.dictionary)
            self.latent_dim = latent_dimension
            self.max_sentence_length = max_sentence_length
            self.seq2seq_model = Seq2SeqWithKeras(self.n_elements, self.latent_dim)

    def compile(self, optimizer='rmsprop', loss='categorical_crossentropy'):
        """Compile the keras model.

        :param optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: rmsprop)
        :param loss: loss function available from keras (Default: 'categorical_crossentropy`)
        """

        # If the model has not been compiled yet, prepare and compile the keras model.
        if not self.compiled:
            self.seq2seq_model.prepare_model()
            self.seq2seq_model.compile(optimizer=optimizer, loss=loss)
            self.compiled = True

    def prepare_training_data(self, text_sequences):
        """Transforming sentence to a sequence of numerical vectors.

        :param text_sequences: a list of text sequences
        :return: rank-3 tensors for encoder input, decoder input, and decoder output
        """
        encoder_input = self.encoder.encode_sentences(text_sequences[:-1], startsig=True, maxlen=self.max_sentence_length, sparse=False)
        decoder_input = self.encoder.encode_sentences(text_sequences[1:], startsig=True, maxlen=self.max_sentence_length, sparse=False)
        decoder_output = self.encoder.encode_sentences(text_sequences[1:], endsig=True, maxlen=self.max_sentence_length, sparse=False)
        return encoder_input, decoder_input, decoder_output

    def train(self, text_sequences, batch_size=64, epochs=100, optimizer='rmsprop', loss='categorical_crossentropy'):
        """Train the character-based seq2seq model.

        :param text_sequences: a list of text sequences
        :param batch_size: batch size (Default: 64)
        :param epochs: number of epochs (Default: 100)
        :param optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: rmsprop)
        :param loss: loss function available from keras (Default: 'categorical_crossentropy`)
        """

        # Prepare the training data.
        encoder_input, decoder_input, decoder_output = self.prepare_training_data(text_sequences)

        # Compile the model.
        self.compile(optimizer=optimizer, loss=loss)

        # Train the model.
        self.seq2seq_model.fit(encoder_input, decoder_input, decoder_output, batch_size=batch_size, epochs=epochs)

    def decode(self, text_sequence, stochastic=True):
        """Given an input text, produce the output text.

        :param text_sequence: input text
        :return: output text
        """

        # Encode the input as state vectors.
        input_vector = np.array([self.encoder.encode_sentence(text_sequence, maxlen=self.max_sentence_length, endsig=True).toarray()])
        states_value = self.seq2seq_model.encoder_model.predict(input_vector)

        # Generate empty target sequence of length 1.
        target_sequence = np.zeros((1, 1, self.n_elements))

        # Populate the first character of target sequence with the start character.
        target_sequence[0, 0, self.dictionary.token2id['\n']] = 1.

        # Sampling loop for a batch of sequences
        stop_condition = False
        decoded_text_sequence = ''
        while not stop_condition:
            output_tokens, h, c = self.seq2seq_model.decoder_model.predict([target_sequence] + states_value)

            # Sample a token
            if stochastic:
                sampled_token_index = np.random.choice(np.arange(output_tokens.shape[2]), p=output_tokens[0, -1, :])
            else:
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.dictionary[sampled_token_index]
            decoded_text_sequence += sampled_char

            # Exit condition: either hit max length or find stop character.
            if sampled_char == '\n' or len(decoded_text_sequence) > self.max_sentence_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_sequence = np.zeros((1, 1, self.n_elements))
            target_sequence[0, 0, sampled_token_index] = 1.

            # Update states.
            states_value = [h, c]

        return decoded_text_sequence

    def save_model(self, file_prefix, final=False):
        """Save the trained models into a single file.

        To save it compactly, call :func:`~save_compact_model`.

        If `final` is set to `True`, the model cannot be further trained.

        If there is no trained model, a `ModelNotTrainedException` will be thrown.

        :param file_prefix: prefix of the file path
        :param final: whether the model is final (that should not be trained further) (Default: False)
        """

        # Save the trained models.
        self.seq2seq_model.save_model(file_prefix, final=final)

        # Save the dictionary.
        self.dictionary.save(file_prefix + '_dictionary.dict')

        # Save the hyperparameters.
        json.dump({'max_sentence_length': self.max_sentence_length, 'latent_dimension': self.latent_dim},
                  open(file_prefix + '_seq2seq.json', 'w'))

    def load_model(self, file_prefix):
        """Load a trained model from multiple files.

        To load a compact model, call :func:`~load_compact_model`.

        :param file_prefix: prefix of the file path
        """

        # Load the dictionary.
        self.dictionary = gensim.corpora.Dictionary.load(file_prefix + '_dictionary.dict')

        # Load the trained model.
        self.seq2seq_model = loadSeq2SeqWithKeras(file_prefix, compact=False)

        # Create an instance of SentenceToCharVecEncoder class.
        self.encoder = SentenceToCharVecEncoder(self.dictionary)

        # Set the remaining parameters.
        self.n_elements = len(self.dictionary)
        hyperparameters = json.load(open(file_prefix + '_seq2seq.json', 'r'))
        self.latent_dim, self.max_sentence_length = hyperparameters['latent_dimension'], hyperparameters['max_sentence_length']
        self.compiled = True


def load_char_based_seq2seq_generator(file_path, compact=True):
    """Load a trained `CharBasedSeq2SeqGenerator` class from file.

    :param file_path: path of the model file
    :param compact: whether it is a compact model (Default: True)
    :return: a `CharBasedSeq2SeqGenerator` class for sequence to sequence inference
    """

    # Create an instance of CharBasedSeq2SeqGenerator class.
    seq2seq_generator = CharBasedSeq2SeqGenerator(None, 0, 0)

    # Load the model from file based on the value of the compact parameter.
    if compact:
        seq2seq_generator.load_compact_model(file_path)
    else:
        seq2seq_generator.load_model(file_path)

    return seq2seq_generator