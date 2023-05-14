import gensim

from .s2skeras import Seq2SeqWithKeras, loadSeq2SeqWithKeras, kerasseq2seq_suffices
from ..charbase.char2vec import SentenceToCharVecEncoder
from shorttext.utils import compactmodel_io as cio


charbases2s_suffices = kerasseq2seq_suffices + ['_dictionary.dict', '_charbases2s.json']


class CharBasedSeq2SeqGenerator(cio.CompactIOMachine):
    """ Class implementing character-based sequence-to-sequence (seq2seq) learning model.

    This class implements the seq2seq model at the character level. This class calls
    :class:`~Seq2SeqWithKeras`.

    Reference:

    Oriol Vinyals, Quoc Le, "A Neural Conversational Model," arXiv:1506.05869 (2015). [`arXiv
    <https://arxiv.org/abs/1506.05869>`_]
    """
    def __init__(self, sentence_encoder=None, latent_size=0, max_length=0):
        """ Instantiate the class.

        If no one-hot encoder passed in, no compilation will be performed.

        :param sentence_encoder: the one-hot encoder
        :param latent_size: number of latent dimensions
        :param max_length: maximum length of a sentence
        :type sentence_encoder: SentenceToCharVecEncoder
        :type latent_size: int
        :type max_length: int
        """
        cio.CompactIOMachine.__init__(self, {'classifier': 'charbases2s'}, 'charbases2s', charbases2s_suffices)
        self.compiled = False
        if sentence_encoder is not None:
            self.sentence_encoder = sentence_encoder
            self.dictionary = self.sentence_encoder.dictionary
            self.vocab_size = len(self.dictionary)
            self.latent_size = latent_size
            self.max_length = max_length
            self.seq2seq_model = Seq2SeqWithKeras(self.vocab_size, self.latent_size)

    def compile(self, optimizer='rmsprop', loss='categorical_crossentropy'):
        """ Compile the keras model.

        :param optimizer: optimizer for gradient descent. Options:
            'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam' (Default: 'rmsprop')
        :param loss: loss function available from keras (Default: 'categorical_crossentropy`)
        :return: None
        :type optimizer: str
        :type loss: str
        """
        if not self.compiled:
            self.seq2seq_model.prepare_model()
            self.seq2seq_model.compile(optimizer=optimizer, loss=loss)
            self.compiled = True

    def prepare_trainingdata(self, text_sequence):
        """ Transform sentence to a sequence of numerical vectors.

        :param text_sequence: text
        :return: rank-3 tensors for encoder input, decoder input, and decoder output
        :type text_sequence: str
        :rtype: (numpy.array, numpy.array, numpy.array)
        """
        encoder_input = self.sentence_encoder.encode_sentences(text_sequence[:-1], startsig=True, maxlen=self.max_length, sparse=False)
        decoder_input = self.sentence_encoder.encode_sentences(text_sequence[1:], startsig=True, maxlen=self.max_length, sparse=False)
        decoder_output = self.sentence_encoder.encode_sentences(text_sequence[1:], endsig=True, maxlen=self.max_length, sparse=False)
        return encoder_input, decoder_input, decoder_output

    def train(self, text_sequence, batch_size=64, epochs=100, optimizer='rmsprop', loss='categorical_crossentropy'):
        """ Train the character-based seq2seq model.

        :param text_sequence: text
        :param batch_size: batch size (Default: 64)
        :param epochs: number of epochs (Default: 100)
        :param optimizer: optimizer for gradient descent. Options:
            'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam' (Default: 'rmsprop')
        :param loss: loss function available from keras (Default: 'categorical_crossentropy`)
        :return: None
        :type text_sequence: str
        :type batch_size: int
        :type epochs: int
        :type optimizer: str
        :type loss: str
        """
        encoder_input, decoder_input, decoder_output = self.prepare_trainingdata(text_sequence)
        self.compile(optimizer=optimizer, loss=loss)
        self.seq2seq_model.fit(encoder_input, decoder_input, decoder_output, batch_size=batch_size, epochs=epochs)

    def decode(self, text_sequence, stochastic=True):
        """ Given an input text, produce the output text.

        :param text_sequence: input text
        :param stochastic: whether to sample output characters (Default: True)
        :return: output text
        :type text_sequence: str
        :type stochastic: bool
        :rtype: str
        """
        # Encode the input as state vectors.
        input_vec = np.array([self.sentence_encoder.encode_sentence(text_sequence, maxlen=self.max_length, endsig=True).toarray()])
        states_value = self.seq2seq_model.encoder_model.predict(input_vec)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.vocab_size))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.dictionary.token2id['\n']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_text_sequence = ''
        while not stop_condition:
            output_tokens, h, c = self.seq2seq_model.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            if stochastic:
                sampled_token_index = np.random.choice(np.arange(output_tokens.shape[2]),
                                                       p=output_tokens[0, -1, :])
            else:
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.dictionary[sampled_token_index]
            decoded_text_sequence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_text_sequence) > self.max_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.vocab_size))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_text_sequence

    def save_model(self, prefix, final=False):
        """ Save the trained models into multiple files.

        To save it compactly, call :func:`~save_compact_model`.

        If `final` is set to `True`, the model cannot be further trained.

        If there is no trained model, a `ModelNotTrainedException` will be thrown.

        :param prefix: prefix of the file path
        :param final: whether the model is final (that should not be trained further) (Default: False)
        :return: None
        :type prefix: str
        :type final: bool
        :raise: ModelNotTrainedException
        """
        self.seq2seq_model.save_model(prefix, final=final)
        self.dictionary.save(prefix+'_dictionary.dict')
        json.dump({'max_length': self.max_length, 'latent_size': self.latent_size}, open(prefix+'_charbases2s.json', 'w'))

    def load_model(self, prefix):
        """ Load a trained model from various files.

        To load a compact model, call :func:`~load_compact_model`.

        :param prefix: prefix of the file path
        :return: None
        :type prefix: str
        """
        self.dictionary = gensim.corpora.Dictionary.load(prefix+'_dictionary.dict')
        self.seq2seq_model = loadSeq2SeqWithKeras(prefix, compact=False)
        self.sentence_encoder = SentenceToCharVecEncoder(self.dictionary)
        self.vocab_size = len(self.dictionary)
        hyperparameters = json.load(open(prefix+'_charbases2s.json', 'r'))
        self.latent_size, self.max_length = hyperparameters['latent_size'], hyperparameters['max_length']
        self.compiled = True

def load_char_based_seq2seq_generator(path, compact=True):
    """ Load a trained `CharBasedSeq2SeqGenerator` class from file.

    :param path: path of the model file
    :param compact: whether it is a compact model (Default: True)
    :return: a `CharBasedSeq2SeqGenerator` class for sequence to sequence inference
    :type path: str
    :type compact: bool
    :rtype: CharBasedSeq2SeqGenerator
    """
    seq2seqer = CharBasedSeq2SeqGenerator(None, 0, 0)
    if compact:
        seq2seqer.load_compact_model(path)
    else:
        seq2seqer.load_model(path)
    return seq2seqer