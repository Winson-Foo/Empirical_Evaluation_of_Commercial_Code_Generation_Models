import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda
from tensorflow.keras.models import Model

from ISR.models.imagemodel import ImageModel


WEIGHTS_URLS = {
    'gans': {
        'arch_params': {'conv_layers': 4, 'rdb_blocks': 3, 'growth_rate': 32, 'init_filters': 32, 'scale': 4, 'rrdb_blocks': 10},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
        'name': 'rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
    },
}
CONV_KERNEL_SIZE = 3
LFF_KERNEL_SIZE = 1


def make_model(arch_params, patch_size):
    """Returns the model used to select the appropriate model."""
    
    return RRDN(arch_params, patch_size)


def get_network(weights):
    """Gets the network parameters from the given weights."""

    if weights in WEIGHTS_URLS.keys():
        arch_params = WEIGHTS_URLS[weights]['arch_params']
        url = WEIGHTS_URLS[weights]['url']
        name = WEIGHTS_URLS[weights]['name']
    else:
        raise ValueError('Available RRDN network weights: {}'.format(list(WEIGHTS_URLS.keys())))

    c_dim = 3
    return arch_params, c_dim, url, name


class RRDN(ImageModel):
    """Implementation of the Residual in Residual Dense Network for image super-scaling.

    The network is the one described in https://arxiv.org/abs/1809.00219 (Wang et al. 2018).

    Args:
        arch_params: dictionary, contains the network parameters conv_layers, rdb_blocks, growth_rate,
            init_filters, scale, and rrdb_blocks.
        patch_size: integer or None, determines the input size. Only needed at training time,
            for prediction is set to None.
        beta: float <= 1, scaling parameter for the residual connections.
        c_dim: integer, number of channels of the input image.
        kernel_size: integer, common kernel size for convolutions.
        upscaling: string, 'ups' or 'shuffle', determines which implementation
            of the upscaling layer to use.
        init_val: extreme values for the RandomUniform initializer.
        weights: string, if not empty, download and load pre-trained weights.
            Overrides other parameters.

    Attributes:
        conv_layers: integer, number of convolutional layers inside each residual dense blocks (RDB).
        rdb_blocks: integer, number of RDBs inside each Residual in Residual Dense Block (RRDB).
        rrdb_blocks: integer, number or RRDBs.
        growth_rate: integer, number of convolution output filters inside the RDBs.
        init_filters: integer, number of output filters of each RDB.
        scale: integer, the scaling factor.
        model: Keras model of the RRDN.
        name: name used to identify what upscaling network is used during training.
        model._name: identifies this network as the generator network
            in the compound model built by the trainer class.
    """
    
    def __init__(
        self,
        arch_params={},
        patch_size=None,
        beta=0.2,
        c_dim=3,
        kernel_size=CONV_KERNEL_SIZE,
        init_val=0.05,
        weights='',
    ):
        if weights:
            arch_params, c_dim, url, fname = get_network(weights)

        self.params = arch_params
        self.beta = beta
        self.c_dim = c_dim
        self.conv_layers = self.params['conv_layers']
        self.rdb_blocks = self.params['rdb_blocks']
        self.growth_rate = self.params['growth_rate']
        self.init_filters = self.params['init_filters']
        self.rrdb_blocks = self.params['rrdb_blocks']
        self.scale = self.params['scale']
        self.initializer = RandomUniform(minval=-init_val, maxval=init_val, seed=None)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.model = self._build_rdn()
        self.model._name = 'generator'
        self.name = 'rrdn'
        if weights:
            weights_path = tf.keras.utils.get_file(fname=fname, origin=url)
            self.model.load_weights(weights_path)
    
    def _dense_block(self, input_layer, block_num, rrdb_num):
        """Implementation of the (Residual) Dense Block as in the paper
        Residual Dense Network for Image Super-Resolution (Zhang et al. 2018).
        
        Args:
            input_layer: input tensor of the dense block.
            block_num: an integer that indicates the number of the dense block.
            rrdb_num: an integer that indicates the number of the Residual in Residual Dense Block.
        Returns:
            A tensor representing the output of the dense block.
        """
        
        x = input_layer
        for conv_num in range(1, self.conv_layers + 1):
            F_dc = Conv2D(
                self.growth_rate,
                kernel_size=self.kernel_size,
                padding='same',
                kernel_initializer=self.initializer,
                name=f'F_{rrdb_num}_{block_num}_{conv_num}',
            )(x)
            F_dc = Activation('relu', name=f'F_{rrdb_num}_{block_num}_{conv_num}_Relu')(F_dc)
            x = concatenate([x, F_dc], axis=3, name=f'RDB_Concat_{rrdb_num}_{block_num}_{conv_num}')
        
        x = Conv2D(
            self.init_filters,
            kernel_size=LFF_KERNEL_SIZE,
            padding='same',
            kernel_initializer=self.initializer,
            name=f'LFF_{rrdb_num}_{block_num}',
        )(x)
        return x
    
    def _RRDB(self, input_layer, rrdb_num):
        """Residual in Residual Dense Block.
        
        Args:
            input_layer: input tensor of the RRDB.
            rrdb_num: an integer that indicates the number of the Residual in Residual Dense Block.
        Returns:
            A tensor representing the output of the RRDB.
        """
        
        x = input_layer
        for block_num in range(1, self.rdb_blocks + 1):
            LFF = self._dense_block(x, block_num, rrdb_num)
            LFF_beta = MultiplyBeta(self.beta)(LFF)
            x = Add(name=f'LRL_{rrdb_num}_{block_num}')([x, LFF_beta])
        
        x = MultiplyBeta(self.beta)(x)
        x = Add(name=f'RRDB_{rrdb_num}_out')([input_layer, x])
        return x
    
    def _pixel_shuffle(self, input_layer):
        """Pixel shuffle implementation of the upscaling part.
        
        Args:
            input_layer: input tensor of the pixel shuffle layer.
        Returns:
            A tensor representing the output of the pixel shuffle layer.
        """
        x = Conv2D(
            self.c_dim * self.scale ** 2,
            kernel_size=CONV_KERNEL_SIZE,
            padding='same',
            kernel_initializer=self.initializer,
            name='PreShuffle',
        )(input_layer)

        return PixelShuffle(self.scale)(x)
    
    def _build_rdn(self):
        """Returns the Keras model of the RRDN."""
        
        LR_input = Input(shape=(self.patch_size, self.patch_size, self.c_dim), name='LR_input')
        pre_blocks = Conv2D(
            self.init_filters,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='Pre_blocks_conv',
        )(LR_input)
        
        for rrdb_num in range(1, self.rrdb_blocks + 1):
            if rrdb_num == 1:
                x = self._RRDB(pre_blocks, rrdb_num)
            else:
                x = self._RRDB(x, rrdb_num)
        
        post_blocks = Conv2D(
            self.init_filters,
            kernel_size=CONV_KERNEL_SIZE,
            padding='same',
            kernel_initializer=self.initializer,
            name='post_blocks_conv',
        )(x)
        GRL = Add(name='GRL')([post_blocks, pre_blocks])
        PS = self._pixel_shuffle(GRL)
        SR = Conv2D(
            self.c_dim,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='SR',
        )(PS)
        
        return Model(inputs=LR_input, outputs=SR)


class PixelShuffle(tf.keras.layers.Layer):
    """Pixel Shuffle layer that performs the upscaling part of the RRDN."""
    
    def __init__(self, scale, *args, **kwargs):
        super(PixelShuffle, self).__init__(*args, **kwargs)
        self.scale = scale