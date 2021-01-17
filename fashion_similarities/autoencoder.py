from tensorflow.keras import losses
from tensorflow.keras.datasets import fashion_mnist
# deep learning modules
from keras.layers import Input, Conv2D, Activation, Dense, Flatten, Reshape, LeakyReLU, \
    BatchNormalization, Conv2DTranspose
from keras.models import Model, Sequential, load_model
from tensorflow.keras import backend as K
from keras.optimizers import Adam

# import callbacks
from keras.callbacks import ModelCheckpoint

# import the necessary packages
import numpy as np
import os


class ConvolutionalAutoencoder(Model):
    @staticmethod
    def build_encoder(width, height, depth, filters, latent_dim):
        input_shape = (height, width, depth)
        channel_dimension = -1
        # start building model
        encoder = Sequential()
        # define the input to the encoder
        encoder.add(Input(shape=input_shape))
        for f in filters:
            encoder.add(Conv2D(f, (3, 3), strides=2, padding="same"))
            encoder.add(LeakyReLU(alpha=0.2))
            encoder.add(BatchNormalization(axis=channel_dimension))
        size_of_last_conv = encoder.layers[-1].input_shape
        encoder.add(Flatten())
        encoder.add(Dense(latent_dim, name="encoded"))
        return encoder, size_of_last_conv

    @staticmethod
    def build_decoder(depth, latent_dim, size, filters):
        channel_dimension = -1
        decoder = Sequential()
        decoder.add(Dense(np.prod(size[1:]), input_shape=(latent_dim,)))
        decoder.add(Reshape((size[1], size[2], size[3])))
        for f in filters[::-1]:
            decoder.add(Conv2DTranspose(f, (3, 3), strides=2, padding="same"))
            decoder.add(LeakyReLU(alpha=0.2))
            decoder.add(BatchNormalization(axis=channel_dimension))
        decoder.add(Conv2DTranspose(depth, (3, 3), padding="same"))
        decoder.add(Activation("sigmoid", name="decoded"))
        return decoder

    def __init__(self, width, height, depth, filters=(32, 64), latent_dim=16):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder, size = self.build_encoder(width, height, depth, filters, latent_dim)
        self.decoder = self.build_decoder(depth, latent_dim, size, filters)

    def call(self, x, training=None, mask=None):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    latent_dim = 64

    x_train = x_train[:10000]

    x_train = x_train.reshape(*x_train.shape, 1)
    x_test = x_test.reshape(*x_test.shape, 1)

    EPOCHS = 1
    INIT_LR = 1e-3
    BS = 64


    model_path = os.path.sep.join(["output", "checkpoints"])

    model_checkpoint = ModelCheckpoint(
        model_path, monitor="val_loss", verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1
    )
    try:
        autoencoder = load_model(model_path)
        print([f"INFO: old learning rate {K.get_value(autoencoder.optimizer.lr)}"])
        K.set_value(autoencoder.optimizer.lr, 1e-3)
        print([f"INFO: new learning rate {K.get_value(autoencoder.optimizer.lr)}"])
    except:
        autoencoder = ConvolutionalAutoencoder(28, 28, 1, latent_dim=latent_dim)
        opt = Adam(learning_rate=INIT_LR)
        autoencoder.compile(optimizer=opt, loss=losses.MeanSquaredError())
    autoencoder.fit(x_train, x_train,
                    epochs=EPOCHS,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[],
                    verbose=1)
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()