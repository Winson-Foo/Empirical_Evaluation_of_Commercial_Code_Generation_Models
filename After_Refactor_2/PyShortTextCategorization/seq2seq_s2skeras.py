import json
from keras.models import load_model, Model
from keras.layers import Input, LSTM, Dense
from shorttext.utils import compactmodel_io as cio
import shorttext.utils.classification_exceptions as e

KERASSEQ2SEQ_SUFFICES = ['.h5', '.json', '_s2s_hyperparam.json', '_encoder.h5', '_encoder.json', '_decoder.h5', '_decoder.json']


class Seq2SeqWithKeras(cio.CompactIOMachine):
    """Class implementing sequence-to-sequence (seq2seq) learning with keras."""
    
    def __init__(self, vec_size, latent_dim):
        """
        Instantiate the class.
        
        Args:
        ----------
        vec_size : int
            vector size of the sequence
        latent_dim : int
            latent dimension in the RNN cell
        """
        super().__init__({'classifier': 'kerasseq2seq'}, 'kerasseq2seq', KERASSEQ2SEQ_SUFFICES)
        self.vec_size = vec_size
        self.latent_dim = latent_dim
        self.compiled = False
        self.trained = False

    def prepare_model(self):
        """Prepare the keras model."""
        encoder_inputs = Input(shape=(None, self.vec_size))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.vec_size))
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.vec_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        self.model, self.encoder_model, self.decoder_model = model, encoder_model, decoder_model
        
    def compile_model(self, optimizer='rmsprop', loss='categorical_crossentropy'):
        """Compile the keras model after preparation running :func:`~prepare_model`.
        
        Args:
        ----------
        optimizer : str
            optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam.
        loss : str
            loss function available from keras
        """
        self.model.compile(optimizer=optimizer, loss=loss)
        self.compiled = True

    def fit_model(self, encoder_input, decoder_input, decoder_output, batch_size=64, epochs=100):
        """Fit the sequence to learn the sequence-to-sequence (seq2seq) model.
        
        Args:
        ----------
        encoder_input : numpy.array
            encoder input, a rank-3 tensor
        decoder_input : numpy.array
            decoder input, a rank-3 tensor
        decoder_output : numpy.array
            decoder output, a rank-3 tensor
        batch_size : int
            batch size
        epochs : int
            number of epochs
        """
        self.model.fit([encoder_input, decoder_input], decoder_output, batch_size=batch_size, epochs=epochs)
        self.trained = True

    def save_model(self, prefix, is_final=False):
        """Save the trained models into multiple files.
        
        Args:
        ----------
        prefix : str
            prefix of the file path
        is_final : bool
            whether the model is final
        """
        if not self.trained:
            raise e.ModelNotTrainedException()

        with open(f"{prefix}_s2s_hyperparam.json", "w") as f:
            json.dump({'vec_size': self.vec_size, 'latent_dim': self.latent_dim}, f)

        if is_final:
            self.model.save_weights(f"{prefix}.h5")
        else:
            self.model.save(f"{prefix}.h5")
        with open(f"{prefix}.json", "w") as f:
            f.write(self.model.to_json())

        if is_final:
            self.encoder_model.save_weights(f"{prefix}_encoder.h5")
            self.decoder_model.save_weights(f"{prefix}_decoder.h5")
        else:
            self.encoder_model.save(f"{prefix}_encoder.h5")
            self.decoder_model.save(f"{prefix}_decoder.h5")
        with open(f"{prefix}_encoder.json", "w") as f:
            f.write(self.encoder_model.to_json())
        with open(f"{prefix}_decoder.json", "w") as f:
            f.write(self.decoder_model.to_json())

    def load_model(self, prefix):
        """Load a trained model from various files.
        
        Args:
        ----------
        prefix : str
            prefix of the file path
        """
        hyperparameters = json.load(open(f"{prefix}_s2s_hyperparam.json", "r"))
        self.vec_size, self.latent_dim = hyperparameters['vec_size'], hyperparameters['latent_dim']
        self.model = load_model(f"{prefix}.h5")
        self.encoder_model = load_model(f"{prefix}_encoder.h5")
        self.decoder_model = load_model(f"{prefix}_decoder.h5")
        self.trained = True


def load_seq2seq_with_keras(path, is_compact=True):
    """Load a trained `Seq2SeqWithKeras` class from file.
    
    Args:
    ----------
    path : str
        path of the model file
    is_compact : bool
        whether it is a compact model
    """
    generator = Seq2SeqWithKeras(0, 0)
    if is_compact:
        generator.load_compact_model(path)
    else:
        generator.load_model(path)
    generator.compiled = True
    return generator