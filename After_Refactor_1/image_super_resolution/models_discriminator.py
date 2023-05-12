from tensorflow.keras.layers import Input, Activation, Dense, Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Discriminator:
    PATCH_SIZE = 48
    KERNEL_SIZE = 3
    FILTERS = (64, 128, 128, 256, 256, 512, 512)
    STRIDES = (2, 1, 2, 1, 1, 1, 1)
    DENSE_SIZE = FILTERS[-1] * 2
    OPTIMIZER = Adam(0.0002, 0.5)
    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy']

    def __init__(self):
        self.model = self._build_model()
        self._compile_model()
        self._set_model_names()

    def _build_conv_block(self, input, filters, strides, use_batch_norm=True, count=None):
        x = Conv2D(
            filters,
            kernel_size=self.KERNEL_SIZE,
            strides=strides,
            padding='same',
            name=f'Conv_{count}',
        )(input)
        x = LeakyReLU(alpha=0.2)(x)
        if use_batch_norm:
            x = BatchNormalization(momentum=0.8)(x)
        return x

    def _build_model(self):
        input_layer = Input(shape=(self.PATCH_SIZE, self.PATCH_SIZE, 3))
        x = self._build_conv_block(input_layer, filters=self.FILTERS[0], strides=1, use_batch_norm=False, count=1)
        for i in range(self.block_num):
            x = self._build_conv_block(
                x,
                filters=self.FILTERS[i],
                strides=self.STRIDES[i],
                count=i + 2,
            )
        x = Dense(self.DENSE_SIZE, name='Dense_1024')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, name='Dense_last')(x)
        output = Activation('sigmoid')(x)
        return Model(inputs=input_layer, outputs=output)

    def _compile_model(self):
        self.model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER, metrics=self.METRICS)

    def _set_model_names(self):
        self.model._name = 'discriminator'
        self.name = 'srgan-large'