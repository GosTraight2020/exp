from tensorflow.keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class DCWGAN:
    def __init__(self,
                 learning_rate_G=None,
                 learning_rate_D=None,
                 batch_size=None,
                 image_size = None,
                 dataset=None,
                 condtional=False):

        self.learning_rate_G = learning_rate_G
        self.learning_rate_D = learning_rate_D
        self.batch_size = batch_size
        self.optimizer_G = Adam(learning_rate_G, 0.5, clipvalue=1.0, decay=1e-8)
        self.optimizer_D = Adam(learning_rate_D, 0.5, clipvalue=1.0, decay=1e-8)
        self.dataset = dataset
        self.image_size = image_size
        self.condtional = condtional
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
    def build_generator(self):
        print('[INFO] Constructing generator model...')
        gen_input = Input(shape=(100, ))

        x = Dense(128 * 16 * 16)(gen_input)
        x = LeakyReLU()(x)
        x = Reshape((16, 16, 128))(x)

        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

        x = Conv2D(512, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(512, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(3, 7, activation='tanh', padding='same')(x)

        x = Reshape((128*128*3,))(x)

        generator = Model(gen_input, x)
        return generator


    def build_discriminator(self):
        print('[INFO] Constructing discriminator model...')
        disc_input = Input(shape=(128* 128* 3,))
        x = Reshape((128, 128, 3))(disc_input)

        x = Conv2D(256, 3)(x)
        x = LeakyReLU()(x)

        x = Conv2D(256, 4, strides=2)(x)
        x = LeakyReLU()(x)

        x = Conv2D(256, 4, strides=2)(x)
        x = LeakyReLU()(x)

        x = Conv2D(256, 4, strides=2)(x)
        x = LeakyReLU()(x)

        x = Conv2D(256, 4, strides=2)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dropout(0.4)(x)

        x = Dense(1, activation=None)(x)
        discriminator = Model(disc_input, x)

        # optimizer = RMSprop(
        #     lr=.0001,
        #     clipvalue=1.0,
        #     decay=1e-8
        # )

        # discriminator.compile(
        #     optimizer=optimizer,
        #     loss='binary_crossentropy'
        # )

        return discriminator
