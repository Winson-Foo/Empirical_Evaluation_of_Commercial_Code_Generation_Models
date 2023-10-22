import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import concatenate, Input, Activation, Add, Conv2D, Lambda
from tensorflow.keras.models import Model

from ISR.models.imagemodel import ImageModel

WEIGHTS_URLS = {
    'gans': {
        'arch_params': {'C': 4, 'D': 3, 'G': 32, 'G0': 32, 'x': 4, 'T': 10},
        'url': 'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/ISR/rrdn-C4-D3-G32-G032-T10-x4-GANS/rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
        'name': 'rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5',
    },
}
DEFAULT_INIT_VAL = 0.05
DEFAULT_PATCH_SIZE = None


def make_model(arch_params, patch_size=DEFAULT_PATCH_SIZE):
    """ Returns the model.

    Used to select the model.

    Args:
        arch_params: dictionary, contains the network parameters.
        patch_size: integer or None, determines the input size.

    Returns:
        an instance of RRDN.
    """
    return RRDN(arch_params, patch_size)


def get_network(weights):
    """Returns the network parameters, url, and filename for the given weights.

    Args:
        weights: string, if not empty, download and load pre-trained weights.

    Returns:
        tuple containing arch_params, c_dim, kernel_size, url, and fname.
    """
    if weights in WEIGHTS_URLS.keys():
        arch_params = WEIGHTS_URLS[weights]['arch_params']
        url = WEIGHTS_URLS[weights]['url']
        name = WEIGHTS_URLS[weights]['name']
    else:
        raise ValueError('Available RRDN network weights: {}'.format(list(WEIGHTS_URLS.keys())))
    c_dim = 3
    kernel_size = 3
    return arch_params, c_dim, kernel_size, url, name


class RRDN(ImageModel):
    """Implementation of the Residual in Residual Dense Network for image super-scaling.

    The network is the one described in https://arxiv.org/abs/1809.00219 (Wang et al. 2018).

    Args:
        arch_params: dictionary, contains the network parameters C, D, G, G0, T, x.
        patch_size: integer or None, determines the input size. Only needed at
            training time, for prediction is set to None.
        beta: float <= 1, scaling parameter for the residual connections.
        c_dim: integer, number of channels of the input image.
        kernel_size: integer, common kernel size for convolutions.
        upscaling: string, 'ups' or 'shuffle', determines which implementation
            of the upscaling layer to use.
        init_val: extreme values for the RandomUniform initializer.
        weights: string, if not empty, download and load pre-trained weights.
            Overrides other parameters.

    Attributes:
        C: integer, number of conv layer inside each residual dense blocks (RDB).
        D: integer, number of RDBs inside each Residual in Residual Dense Block (RRDB).
        T: integer, number or RRDBs.
        G: integer, number of convolution output filters inside the RDBs.
        G0: integer, number of output filters of each RDB.
        x: integer, the scaling factor.
        model: Keras model of the RRDN.
        name: name used to identify what upscaling network is used during training.
        model._name: identifies this network as the generator network
            in the compound model built by the trainer class.
    """
    
    def __init__(self, arch_params={}, patch_size=DEFAULT_PATCH_SIZE, beta=0.2,
                 c_dim=3, kernel_size=3, init_val=DEFAULT_INIT_VAL, weights=''):
        """Initializes the RRDN class.

        Args:
            arch_params: dictionary, contains the network parameters.
            patch_size: integer or None, determines the input size.
            beta: float <= 1, scaling parameter for the residual connections.
            c_dim: integer, number of channels of the input image.
            kernel_size: integer, common kernel size for convolutions.
            init_val: extreme values for the RandomUniform initializer.
            weights: string, if not empty, download and load pre-trained weights.
                     Overrides other parameters.
        """
        if weights:
            arch_params, c_dim, kernel_size, url, fname = get_network(weights)

        self.params = arch_params
        self.beta = beta
        self.c_dim = c_dim
        self.C = self.params['C']
        self.D = self.params['D']
        self.G = self.params['G']
        self.G0 = self.params['G0']
        self.T = self.params['T']
        self.scale = self.params['x']
        self.initializer = RandomUniform(minval=-init_val, maxval=init_val, seed=None)
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.model = self._build_rdn()
        self.model._name = 'generator'
        self.name = 'rrdn'
        if weights:
            weights_path = tf.keras.utils.get_file(fname=fname, origin=url)
            self.model.load_weights(weights_path)

    def _dense_block(self, input_layer, d, t):
        """Implementation of the (Residual) Dense Block as in the paper
        Residual Dense Network for Image Super-Resolution (Zhang et al. 2018).
        Residuals are incorporated in the RRDB.
        d is an integer only used for naming. (d-th block)

        Args:
            input_layer: the input layer.
            d: the index of the dense block.
            t: the index of the RRDB.

        Returns:
            the output layer of the dense block.
        """
        x = input_layer
        for c in range(1, self.C + 1):
            F_dc = Conv2D(
                self.G,
                kernel_size=self.kernel_size,
                padding='same',
                kernel_initializer=self.initializer,
                name=f'F_{t}_{d}_{c}',
            )(x)
            F_dc = Activation('relu', name=f'F_{t}_{d}_{c}_Relu')(F_dc)
            x = concatenate([x, F_dc], axis=3, name=f'RDB_Concat_{t}_{d}_{c}')
        x = Conv2D(
            self.G0,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name=f'LFF_{t}_{d}',
        )(x)
        return x

    def _RRDB(self, input_layer, t):
        """Residual in Residual Dense Block.

        t is integer, for naming of RRDB.
        beta is scalar.

        Args:
            input_layer: the input layer.
            t: the index of the RRDB network.

        Returns:
            the output layer of the RRDB.
        """
        x = input_layer
        for d in range(1, self.D + 1):
            LFF = self._dense_block(x, d, t)
            LFF_beta = MultiplyBeta(self.beta)(LFF)
            x = Add(name=f'LRL_{t}_{d}')(inputs=[x, LFF_beta])
        x = MultiplyBeta(self.beta)(x)
        x = Add(name=f'RRDB_{t}_out')(inputs=[input_layer, x])
        return x
        
    def _pixel_shuffle(self, input_layer):
        """PixelShuffle implementation of the upscaling part."""
        x = Conv2D(
            self.c_dim * self.scale ** 2,
            kernel_size=3,
            padding='same',
            kernel_initializer=self.initializer,
            name='PreShuffle',
        )(input_layer)
        return PixelShuffle(self.scale)(x)

    def _build_rdn(self):
        """Builds the RDN network.

        Returns:
            the RDN model.
        """
        LR_input = Input(shape=(self.patch_size, self.patch_size, 3), name='LR_input')
        pre_blocks = Conv2D(
            self.G0,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.initializer,
            name='Pre_blocks_conv',
        )(LR_input)
        for t in range(1, self.T + 1):
            if t == 1:
                x = self._RRDB(pre_blocks, t)
            else:
                x = self._RRDB(x, t)
        post_blocks = Conv2D(
            self.G0,
            kernel_size=3,
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