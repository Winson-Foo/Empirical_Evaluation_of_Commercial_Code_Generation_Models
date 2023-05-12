import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda
from tensorflow.keras.models import Model

from ISR.models.imagemodel import ImageModel

WEIGHTS_URLS = {
    'gans': {
        'arch_params': {'num_conv_layers': 4, 'num_dense_blocks': 3, 'num_filters': 32, 'num_filters_1st_layer': 32, 'scale_factor': 4, 'num_rrdbs': 10},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
        'name': 'rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
    },
}

class PixelShuffle(tf.keras.layers.Layer):
    """
    Pixel shuffle implementation for upscaling part.
    """
    def __init__(self, scale_factor, *args, **kwargs):
        super(PixelShuffle, self).__init__(*args, **kwargs)
        self.scale_factor = scale_factor

    def call(self, x):
        return tf.nn.depth_to_space(x, block_size=self.scale_factor, data_format='NHWC')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale_factor': self.scale_factor,
        })
        return config

class MultiplyBeta(tf.keras.layers.Layer):
    """
    Multiply beta value with input tensor.
    """
    def __init__(self, beta, *args, **kwargs):
        super(MultiplyBeta, self).__init__(*args, **kwargs)
        self.beta = beta

    def call(self, x, **kwargs):
        return x * self.beta

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'beta': self.beta,
        })
        return config

class RRDN(ImageModel):
    """
    Implementation of Residual in Residual Dense Network for image super-scaling.
    """
    def __init__(
            self, arch_params={}, patch_size=None, beta=0.2, num_channels=3, kernel_size=3, init_val=0.05, weights=''
    ):
        """
        Initializes the RRDN model.
        """
        if weights:
            arch_params, num_channels, kernel_size, url, fname = get_network(weights)

        self.params = arch_params
        self.beta = beta
        self.num_channels = num_channels
        self.num_conv_layers = self.params['num_conv_layers']
        self.num_dense_blocks = self.params['num_dense_blocks']
        self.num_filters = self.params['num_filters']
        self.num_filters_1st_layer = self.params['num_filters_1st_layer']
        self.num_rrdbs = self.params['num_rrdbs']
        self.scale_factor = self.params['scale_factor']
        self.initializer = RandomUniform(minval=-init_val, maxval=init_val, seed=None)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.model = self._build_rrdn()
        self.model._name = 'generator'
        self.name = 'rrdn'
        if weights:
            weights_path = tf.keras.utils.get_file(fname=fname, origin=url)
            self.model.load_weights(weights_path)

    def _dense_block(self, input_tensor, block_no, rrdb_no):
        """
        Implements a residual dense block with residual connections.
        """
        x = input_tensor
        for conv_layer_no in range(1, self.num_conv_layers + 1):
            conv_layer = Conv2D(
                self.num_filters,
                kernel_size=self.kernel_size,
                padding='same',
                kernel_initializer=self.initializer,
                name=f'F_{rrdb_no}_{block_no}_{conv_layer_no}',
            )(x)
            conv_layer = Activation('relu', name=f'F_{rrdb_no}_{block_no}_{conv_layer_no}_Relu')(conv_layer)
            x = concatenate([x, conv_layer], axis=3, name=f'RDB_Concat_{rrdb_no}_{block_no}_{conv_layer_no}')

        x = Conv2D(
            self.num_filters_1st_layer,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name=f'LFF_{rrdb_no}_{block_no}',
        )(x)
        return x

    def _rrdb(self, input_tensor, rrdb_no):
        """
        Implements a Residual in Residual Dense Block.
        """
        x = input_tensor
        for block_no in range(1, self.num_dense_blocks + 1):
            dense_block = self._dense_block(x, block_no, rrdb_no)
            dense_block_beta = MultiplyBeta(self.beta)(dense_block)
            x = Add(name=f'LRL_{rrdb_no}_{block_no}')([x, dense_block_beta])

        x = MultiplyBeta(self.beta)(x)
        x = Add(name=f'RRDB_{rrdb_no}_out')([input_tensor, x])
        return x

    def _pixel_shuffle(self, input_tensor):
        """
        Implements the upscaling part using pixel shuffle.
        """
        x = Conv2D(
            self.num_channels * self.scale_factor ** 2,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='PreShuffle',
        )(input_tensor)

        return PixelShuffle(self.scale_factor)(x)

    def _build_rrdn(self):
        """
        Builds the entire RRDN model.
        """
        lr_input = Input(shape=(self.patch_size, self.patch_size, 3), name='LR_input')
        pre_blocks = Conv2D(
            self.num_filters_1st_layer,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='Pre_blocks_conv',
        )(lr_input)

        for rrdb_no in range(1, self.num_rrdbs + 1):
            if rrdb_no == 1:
                x = self._rrdb(pre_blocks, rrdb_no)
            else:
                x = self._rrdb(x, rrdb_no)

        post_blocks = Conv2D(
            self.num_filters_1st_layer,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='post_blocks_conv',
        )(x)
        grl = Add(name='GRL')([post_blocks, pre_blocks])
        ps = self._pixel_shuffle(grl)
        sr_image = Conv2D(
            self.num_channels,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='SR',
        )(ps)
        return Model(inputs=lr_input, outputs=sr_image)

def get_network(weights):
    """
    Returns network parameters for the given weights.
    """
    if weights in WEIGHTS_URLS.keys():
        arch_params = WEIGHTS_URLS[weights]['arch_params']
        url = WEIGHTS_URLS[weights]['url']
        fname = WEIGHTS_URLS[weights]['name']
    else:
        raise ValueError('Available RRDN network weights: {}'.format(list(WEIGHTS_URLS.keys())))

    num_channels = 3
    kernel_size = 3
    return arch_params, num_channels, kernel_size, url, fname

def make_model(arch_params, patch_size):
    """
    Returns the RRDN model with the given architecture parameters and patch size.
    """
    return RRDN(arch_params, patch_size)