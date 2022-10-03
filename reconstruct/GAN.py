import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Reshape, UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten
class MGAN:
    def __init__(self,
                 model_shape_G=None,
                 model_shape_D=None,
                 learning_rate_G=None,
                 learning_rate_D=None,
                 batch_size=None,
                 X_train=None,
                 y_train=None,
                 X_test=None,
                 y_test=None):
        if model_shape_G is None:
            raise ValueError("[Error] Model shape of G is None!")
        if model_shape_D is None:
            raise ValueError("[Error] Model shape of D is None!")
        if learning_rate_G is None:
            raise ValueError("[Error] learning_rate of G is None!")
        if learning_rate_D is None:
            raise ValueError("[Error] learning_rate of D is None!")
        if X_train is None:
            raise ValueError("[Error] X_train is None!")
        if y_train is None:
            raise ValueError("[Error] y_train is None!")
        if X_test is None:
            raise ValueError("[Error] X_test is None!")
        if y_test is None:
            raise ValueError("[Error] y_test is None!")
        if batch_size is None:
            raise ValueError("[Error] batch_size is None!")

        self.model_shape_G = model_shape_G
        self.model_shape_D = model_shape_D
        self.learning_rate_G = learning_rate_G
        self.learning_rate_D = learning_rate_D
        self.batch_size = batch_size
        self.optimizer_G = tf.train.AdamOptimizer(learning_rate_G, 0.5)
        self.optimizer_D = tf.train.AdamOptimizer(learning_rate_D, 0.5)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.generator = self.build_generator(input_shape=self.model_shape_G[0])
        self.discriminator = self.build_discriminator(input_shape=self.model_shape_D[0])

        # 打印模型信息
        print("[INFO] Creating Model:\n \
                model_shape_G is :{},\n \
                model_shape_D is :{},\n \
                learning_rate_G is : {}, \n \
                learning_rate_D is : {}, \n \
                batch_size is : {}, \n \
                X_train's shape is : {}, \n \
                y_train's shape is : {}, \n \
                X_test's shape is : {}, \n \
                y_test's shape is : {}".format(self.model_shape_G, self.model_shape_D, self.learning_rate_G, self.learning_rate_D, self.batch_size,  self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape))
        
    def build_generator(self, input_shape=None):
        print('[INFO] Constructing generator model...')
        if input_shape is None:
            raise ValueError("[ERROR] The input shape of generator is None!")
        model_size_G = len(self.model_shape_G)
        input = Input(shape=input_shape)
        model = Sequential()
        model.add(input)
        for i in range(1, model_size_G):
            model.add(Dense(self.model_shape_G[i], activation='sigmoid' if i==model_size_G-1 else 'relu'))
            if i != model_size_G-1:
                model.add(BatchNormalization())
        return model


    def build_discriminator(self, input_shape=None):
        print('[INFO] Constructing discriminator model...')
        if input_shape is None:
            raise ValueError('[Error] The input shape of discriminator is None!')

        model_size_D = len(self.model_shape_D)
        model = Sequential()
        input = Input(shape=input_shape)
        model.add(input)

        for i in range(1, model_size_D):
            model.add(Dense(self.model_shape_D[i], activation=None if i==model_size_D-1 else 'relu'))
            if i != model_size_D-1:
                model.add(BatchNormalization())
        return model



class DCMGAN:
    def __init__(self,
                 input_shape_G=None,
                 input_shape_D=None,
                 learning_rate_G=None,
                 learning_rate_D=None,
                 batch_size=None,
                 X_train=None,
                 y_train=None,
                 X_test=None,
                 y_test=None):
        if input_shape_G is None:
            raise ValueError("[Error] input shape of G is None!")
        if input_shape_D is None:
            raise ValueError("[Error] input shape of D is None!")
        if learning_rate_G is None:
            raise ValueError("[Error] learning_rate of G is None!")
        if learning_rate_D is None:
            raise ValueError("[Error] learning_rate of D is None!")
        if X_train is None:
            raise ValueError("[Error] X_train is None!")
        if y_train is None:
            raise ValueError("[Error] y_train is None!")
        if X_test is None:
            raise ValueError("[Error] X_test is None!")
        if y_test is None:
            raise ValueError("[Error] y_test is None!")
        if batch_size is None:
            raise ValueError("[Error] batch_size is None!")

        self.input_shape_G = input_shape_G
        self.input_shape_D = input_shape_D
        self.learning_rate_G = learning_rate_G
        self.learning_rate_D = learning_rate_D
        self.batch_size = batch_size
        self.optimizer_G = tf.train.AdamOptimizer(learning_rate_G, 0.5)
        self.optimizer_D = tf.train.AdamOptimizer(learning_rate_D, 0.5)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.generator = self.build_generator(input_shape=self.input_shape_G)
        self.discriminator = self.build_discriminator(input_shape=self.input_shape_D)

        # 打印模型信息
        print("[INFO] Creating Model:\n \
                model_shape_G is :{},\n \
                model_shape_D is :{},\n \
                learning_rate_G is : {}, \n \
                learning_rate_D is : {}, \n \
                batch_size is : {}, \n \
                X_train's shape is : {}, \n \
                y_train's shape is : {}, \n \
                X_test's shape is : {}, \n \
                y_test's shape is : {}".format(self.input_shape_G, self.input_shape_D, self.learning_rate_G, self.learning_rate_D, self.batch_size,  self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape))
        
    def build_generator(self, input_shape=None):
        print('[INFO] Constructing generator model...')
        if input_shape is None:
            raise ValueError("[ERROR] The input shape of generator is None!")
        input = Input(shape=input_shape)
        model = Sequential()
        model.add(input)
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(1, (5, 5), padding='same'))
        # model.add(Activation('relu'))
        return model


    def build_discriminator(self, input_shape=None):
        print('[INFO] Constructing discriminator model...')
        print("[DEBUG] Input shape of discriminator is {}".format(input_shape))

        if input_shape is None:
            raise ValueError('[Error] The input shape of discriminator is None!')

        input = Input(shape=input_shape)



        model = Sequential()
        model.add(input)
        model.add(
                Conv2D(64, (5, 5),
                padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('relu'))
        return model