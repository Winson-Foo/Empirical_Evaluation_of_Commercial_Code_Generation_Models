from keras.layers import Input, Conv2D
from keras.models import Model

class Discriminator:
    def __init__(self, patch_size, kernel_size):
        input_hr = Input((patch_size, patch_size, 3))
        conv1 = Conv2D(64, kernel_size, activation='relu', padding='same')(input_hr)
        conv2 = Conv2D(64, kernel_size, activation='relu', padding='same', strides=2)(conv1)
        conv3 = Conv2D(128, kernel_size, activation='relu', padding='same')(conv2)
        conv4 = Conv2D(128, kernel_size, activation='relu', padding='same', strides=2)(conv3)
        conv5 = Conv2D(256, kernel_size, activation='relu', padding='same')(conv4)
        conv6 = Conv2D(256, kernel_size, activation='relu', padding='same', strides=2)(conv5)
        conv7 = Conv2D(512, kernel_size, activation='relu', padding='same')(conv6)
        conv8 = Conv2D(512, kernel_size, activation='relu', padding='same', strides=2)(conv7)
        flat = Flatten()(conv8)
        dense1 = Dense(1024, activation='relu')(flat)
        dense2 = Dense(1, activation='sigmoid')(dense1)
        self.model = Model(inputs=input_hr, outputs=dense2)

    def compile(self, **kwargs):
        self.model.compile(**kwargs)