from tensorflow.keras.layers import Input, Activation, Dense, Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class Discriminator:
    """
    Implementation of the discriminator network for the adversarial
    component of the perceptual loss.
    """

    ALPHA = 0.2
    MOMENTUM = 0.8
    LEARNING_RATE = 0.0002
    BETA_1 = 0.5

    def __init__(self, patch_size, kernel_size=3):
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.block_param = {
            'filters': (64, 128, 128, 256, 256, 512, 512),
            'strides': (2, 1, 2, 1, 1, 1, 1)
        }
        self.block_num = len(self.block_param['filters'])
        self.model = self._build_discriminator()
        optimizer = Adam(self.LEARNING_RATE, self.BETA_1)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model._name = 'discriminator'
        self.name = 'srgan-large'

    def _conv_block(self, input, filters, strides, use_batch_norm=True, count=None):
        """
        This function applies a convolutional layer, followed by a leaky ReLU activation and optional batch normalization.
        """
        x = Conv2D(
            filters,
            kernel_size=self.kernel_size,
            strides=strides,
            padding='same',
            name='Conv_{}'.format(count),
        )(input)
        x = LeakyReLU(alpha=self.ALPHA)(x)
        if use_batch_norm:
            x = BatchNormalization(momentum=self.MOMENTUM)(x)
        return x

    def _dense_layers(self, input, units, count=None):
        """
        This function applies a fully connected layer with a given number of units.
        """
        x = Dense(units, name='Dense_{}'.format(count))(input)
        x = LeakyReLU(alpha=self.ALPHA)(x)
        return x

    def _build_discriminator(self):
        """
        This function puts the discriminator's layers together.
        """
        hr_input = Input(shape=(self.patch_size, self.patch_size, 3))

        x = self._conv_block(hr_input, filters=64, strides=1, use_batch_norm=False, count=1)

        for i in range(self.block_num):
            x = self._conv_block(
                x,
                filters=self.block_param['filters'][i],
                strides=self.block_param['strides'][i],
                count=i + 2,
            )

        x = self._dense_layers(x, units=self.block_param['filters'][-1] * 2, count=1024)
        x = self._dense_layers(x, units=1, count='last')
        sr_output = Activation('sigmoid', name='SR_Output')(x)

        return Model(inputs=hr_input, outputs=sr_output)