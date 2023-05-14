import json

from keras.models import load_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense

from shorttext.utils import compactmodel_io as cio
import shorttext.utils.classification_exceptions as e


class Seq2SeqWithKeras(cio.CompactIOMachine):
    """ Class implementing sequence-to-sequence (seq2seq) learning with keras. """
    def __init__(self, vecsize, latent_dim):
        """ Instantiate the class.

        :param vecsize: vector size of the sequence
        :param latent_dim: latent dimension in the RNN cell
        :type vecsize: int
        :type latent_dim: int
        """
        cio.CompactIOMachine.__init__(self, {'classifier': 'kerasseq2seq'}, 'kerasseq2seq', kerasseq2seq_suffices)
        self.vecsize = vecsize
        self.latent_dim = latent_dim
        self.compiled = False
        self.trained = False

    def prepare_encoder(self, encoder_inputs):
        """ Define encoder layer and get encoder states.

        :param encoder_inputs: input tensor to the encoder
        :type encoder_inputs: keras.layers.Input
        :return: encoder states
        :rtype: list
        """
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        return encoder_states

    def prepare_decoder(self, decoder_inputs, encoder_states):
        """ Define decoder layer and get decoder outputs.

        :param decoder_inputs: input tensor to the decoder
        :param encoder_states: encoder states to initialize the decoder
        :type decoder_inputs: keras.layers.Input
        :type encoder_states: list
        :return: decoder outputs
        :rtype: keras.layers.Dense
        """
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.vecsize, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        return decoder_outputs

    def prepare_sampling_models(self, encoder_inputs, decoder_inputs, decoding_layers):
        """ Define models for encoding, decoding, and sampling.

        :param encoder_inputs: input tensor to the encoder
        :param decoder_inputs: input tensor to the decoder
        :param decoding_layers: layers of the decoder model
        :type encoder_inputs: keras.layers.Input
        :type decoder_inputs: keras.layers.Input
        :type decoding_layers: list
        :return: models for encoding, decoding, and sampling
        :rtype: tuple
        """
        encoder_model = Model(encoder_inputs, decoding_layers[0])
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoding_layers[1](
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoding_layers[2](decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)
        return encoder_model, decoder_model

    def prepare_model(self):
        """ Prepare the keras model.

        :return: None
        """
        encoder_inputs = Input(shape=(None, self.vecsize))
        encoder_states = self.prepare_encoder(encoder_inputs)

        decoder_inputs = Input(shape=(None, self.vecsize))
        decoder_outputs = self.prepare_decoder(decoder_inputs, encoder_states)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        encoding_layers = model.layers[:2]
        decoding_layers = [layer for layer in model.layers[2:]]
        self.encoder_model, self.decoder_model = self.prepare_sampling_models(encoder_inputs, decoder_inputs, decoding_layers)
        self.model = model

    def compile(self, optimizer='rmsprop', loss='categorical_crossentropy'):
        """ Compile the keras model after preparation running :func:`~prepare_model`.

        :param optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: rmsprop)
        :param loss: loss function available from keras (Default: 'categorical_crossentropy`)
        :type optimizer: str
        :type loss: str
        :return: None
        """
        self.model.compile(optimizer=optimizer, loss=loss)
        self.compiled = True

    def fit(self, encoder_input, decoder_input, decoder_output, batch_size=64, epochs=100):
        """ Fit the sequence to learn the sequence-to-sequence (seq2seq) model.

        :param encoder_input: encoder input, a rank-3 tensor
        :param decoder_input: decoder input, a rank-3 tensor
        :param decoder_output: decoder output, a rank-3 tensor
        :param batch_size: batch size (Default: 64)
        :param epochs: number of epochs (Default: 100)
        :return: None
        :type encoder_input: numpy.array
        :type decoder_input: numpy.array
        :type decoder_output: numpy.array
        :type batch_size: int
        :type epochs: int
        """
        self.model.fit([encoder_input, decoder_input], decoder_output,
                       batch_size=batch_size,
                       epochs=epochs)
        self.trained = True

    def savemodel(self, prefix, final=False):
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
        if not self.trained:
            raise e.ModelNotTrainedException()

        json.dump({'vecsize': self.vecsize, 'latent_dim': self.latent_dim}, open(prefix+'_s2s_hyperparam.json', 'w'))

        if final:
            self.model.save_weights(prefix+'.h5')
        else:
            self.model.save(prefix+'.h5')
        open(prefix+'.json', 'w').write(self.model.to_json())

        if final:
            self.encoder_model.save_weights(prefix+'_encoder.h5')
            self.decoder_model.save_weights(prefix + '_decoder.h5')
        else:
            self.encoder_model.save(prefix + '_encoder.h5')
            self.decoder_model.save(prefix+'_decoder.h5')
        open(prefix+'_encoder.json', 'w').write(self.encoder_model.to_json())
        open(prefix+'_decoder.json', 'w').write(self.decoder_model.to_json())

    def loadmodel(self, prefix):
        """ Load a trained model from various files.

        To load a compact model, call :func:`~load_compact_model`.

        :param prefix: prefix of the file path
        :return: None
        :type prefix: str
        """
        hyperparameters = json.load(open(prefix+'_s2s_hyperparam.json', 'r'))
        self.vecsize, self.latent_dim = hyperparameters['vecsize'], hyperparameters['latent_dim']
        self.model = load_model(prefix+'.h5')
        self.encoder_model = load_model(prefix+'_encoder.h5')
        self.decoder_model = load_model(prefix+'_decoder.h5')
        self.trained = True


class ModelIO:
    """ Helper class for saving and loading model files """
    @staticmethod
    def load_seq2seq_with_keras(path, compact=True):
        """ Load a trained `Seq2SeqWithKeras` class from file.

        :param path: path of the model file
        :param compact: whether it is a compact model (Default: True)
        :return: a `Seq2SeqWithKeras` class for sequence to sequence inference
        :type path: str
        :type compact: bool
        :rtype: Seq2SeqWithKeras
        """
        generator = Seq2SeqWithKeras(0, 0)
        if compact:
            generator.load_compact_model(path)
        else:
            generator.loadmodel(path)
        generator.compiled = True
        return generator

    @staticmethod
    def save_seq2seq_with_keras(model, prefix, final=False):
        """ Save the trained `Seq2SeqWithKeras` into multiple files.

        :param model: trained `Seq2SeqWithKeras` instance
        :param prefix: file name prefix
        :param final: whether the model is final and should not be trained further (Default: False)
        :type model: Seq2SeqWithKeras
        :type prefix: str
        :type final: bool
        """
        model.savemodel(prefix, final=final)