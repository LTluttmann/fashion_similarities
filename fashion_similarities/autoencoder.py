# deep learning modules
from keras.layers import (
    Input,
    Conv2D,
    Activation,
    Dense,
    Flatten,
    Reshape,
    PReLU,
    BatchNormalization,
    GaussianNoise,
    MaxPooling2D,
    UpSampling2D
)
from keras.models import Model
from keras.initializers import Constant
from tensorflow.keras import backend as K

# import the necessary packages
import numpy as np
from fashion_similarities.custom_layers import (
    TiedConv2DTranspose,
    DenseTied
)


class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth):
        STRIDE = 2
        POOLING = False
        BIAS = True
        assert POOLING + STRIDE == 2

        inputs = Input(shape=(width, height, depth))
        x = GaussianNoise(0.1)(inputs)

        conv_1 = Conv2D(32, (3, 3), padding="same", strides=1, use_bias=BIAS)
        x = conv_1(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        conv_2 = Conv2D(32, (3, 3), padding="same", strides=STRIDE, use_bias=BIAS)
        x = conv_2(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        if POOLING:
            x = MaxPooling2D((2, 2))(x)

        conv_3 = Conv2D(64, (3, 3), padding="same", strides=1, use_bias=BIAS)
        x = conv_3(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        conv_4 = Conv2D(64, (3, 3), padding="same", strides=STRIDE, use_bias=BIAS)
        x = conv_4(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        if POOLING:
            x = MaxPooling2D((2, 2))(x)

        conv_5 = Conv2D(128, (3, 3), padding="same", strides=1, use_bias=BIAS)
        x = conv_5(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        conv_6 = Conv2D(128, (3, 3), padding="same", strides=STRIDE, use_bias=BIAS)
        x = conv_6(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        if POOLING:
            x = MaxPooling2D((2, 2))(x)

        conv_7 = Conv2D(256, (3, 3), padding="same", strides=1, use_bias=BIAS)
        x = conv_7(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        conv_8 = Conv2D(256, (3, 3), padding="same", strides=STRIDE, use_bias=BIAS)
        x = conv_8(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        if POOLING:
            x = MaxPooling2D((2, 2))(x)

        conv_9 = Conv2D(512, (3, 3), padding="same", strides=1, use_bias=BIAS)
        x = conv_9(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        conv_10 = Conv2D(512, (3, 3), padding="same", strides=STRIDE, use_bias=BIAS)
        x = conv_10(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        if POOLING:
            x = MaxPooling2D((2, 2))(x)

        # flatten the network and then construct the latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(1024, name="encoded")
        x = latent(x)
        x = DenseTied(np.prod(volumeSize[1:]), tied_to=latent)(x)

        # start building the decoder model which will accept the
        # output of the encoder as its inputs
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        if POOLING:
            x = UpSampling2D()(x)

        x = TiedConv2DTranspose(512, (3, 3), padding="same", tied_to=conv_10, strides=STRIDE, use_bias=BIAS)(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        x = TiedConv2DTranspose(256, (3, 3), padding="same", tied_to=conv_9, strides=1, use_bias=BIAS)(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        if POOLING:
            x = UpSampling2D()(x)

        x = TiedConv2DTranspose(256, (3, 3), padding="same", tied_to=conv_8, strides=STRIDE, use_bias=BIAS)(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        x = TiedConv2DTranspose(128, (3, 3), padding="same", tied_to=conv_7, strides=1, use_bias=BIAS)(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        if POOLING:
            x = UpSampling2D()(x)

        x = TiedConv2DTranspose(128, (3, 3), padding="same", tied_to=conv_6, strides=STRIDE, use_bias=BIAS)(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        x = TiedConv2DTranspose(64, (3, 3), padding="same", tied_to=conv_5, strides=1, use_bias=BIAS)(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        if POOLING:
            x = UpSampling2D()(x)

        x = TiedConv2DTranspose(64, (3, 3), padding="same", tied_to=conv_4, strides=STRIDE, use_bias=BIAS)(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        x = TiedConv2DTranspose(32, (3, 3), padding="same", tied_to=conv_3, strides=1, use_bias=BIAS)(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        if POOLING:
            x = UpSampling2D()(x)

        x = TiedConv2DTranspose(32, (3, 3), padding="same", tied_to=conv_2, strides=STRIDE, use_bias=BIAS)(x)
        x = BatchNormalization()(x)
        x = PReLU(alpha_initializer=Constant(value=0.25))(x)

        x = TiedConv2DTranspose(3, (3, 3), padding="same", tied_to=conv_1, strides=1, use_bias=BIAS)(x)
        x = BatchNormalization()(x)
        outputs = Activation("sigmoid")(x)
        ae = Model(inputs=inputs, outputs=outputs)
        return ae


if __name__ == "__main__":
    ae = ConvAutoencoder.build(224, 224, 3)
    print(ae.summary())
