import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Reshape, UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Conv2DTranspose, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
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
                 learning_rate_G=None,
                 learning_rate_D=None,
                 batch_size=None,
                 nc = None,
                 nz = None,
                 ngf = None,
                 ndf = None,
                 n_extra_layers = None,
                 Diters = None,
                 image_size = None):

        self.learning_rate_G = learning_rate_G
        self.learning_rate_D = learning_rate_D
        self.batch_size = batch_size
        self.optimizer_G = tf.train.AdamOptimizer(learning_rate_G, 0.5)
        self.optimizer_D = tf.train.AdamOptimizer(learning_rate_D, 0.5)
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ndf = ndf
        self.n_extra_layers = n_extra_layers
        self.Diters = Diters
        self.image_size = image_size
        self.generator = self.build_generator(self.image_size, self.nz, self.nc, self.ngf, self.n_extra_layers)
        self.discriminator = self.build_discriminator(self.image_size, self.nz, self.nc, self.ndf, self.n_extra_layers)
        
        # 打印模型信息
        print("[INFO] Creating Model:\n \
                learning_rate_G is : {}, \n \
                learning_rate_D is : {}, \n \
                batch_size is : {}".format(self.learning_rate_G, self.learning_rate_D, self.batch_size))
        
    def build_generator(self, isize, nz, nc, ngf, n_extra_layers=0):
        print('[INFO] Constructing generator model...')
        print("generator:{}, {}, {}, {}, {}".format(isize, nz, nc, ngf, n_extra_layers))
        conv_init = RandomNormal(0, 0.02)
        gamma_init = RandomNormal(1., 0.02)
        cngf= ngf//2
        tisize = isize
        while tisize > 5:
            cngf = cngf * 2
            assert tisize%2==0
            tisize = tisize // 2
        _ = inputs = Input(shape=(nz,))
        _ = Reshape((1,1, nz))(_)
        _ = Conv2DTranspose(filters=cngf, kernel_size=tisize, strides=1, use_bias=False,
                            kernel_initializer = conv_init, 
                            name = 'initial.{0}-{1}.convt'.format(nz, cngf))(_)
        _ = BatchNormalization(gamma_initializer = gamma_init, momentum=0.9, axis=-1, epsilon=1.01e-5,
                                name = 'initial.{0}.batchnorm'.format(cngf))(_, training=1)
        _ = Activation("relu", name = 'initial.{0}.relu'.format(cngf))(_)
        csize, cndf = tisize, cngf
        

        while csize < isize//2:
            in_feat = cngf
            out_feat = cngf//2
            _ = Conv2DTranspose(filters=out_feat, kernel_size=4, strides=2, use_bias=False,
                            kernel_initializer = conv_init, padding="same",
                            name = 'pyramid.{0}-{1}.convt'.format(in_feat, out_feat)             
                            ) (_)
            _ = BatchNormalization(gamma_initializer = gamma_init, 
                                    momentum=0.9, axis=-1, epsilon=1.01e-5,
                                    name = 'pyramid.{0}.batchnorm'.format(out_feat))(_, training=1)
            
            _ = Activation("relu", name = 'pyramid.{0}.relu'.format(out_feat))(_)
            csize, cngf = csize*2, cngf//2
        _ = Conv2DTranspose(filters=nc, kernel_size=4, strides=2, use_bias=False,
                            kernel_initializer = conv_init, padding="same",
                            name = 'final.{0}-{1}.convt'.format(cngf, nc)
                            )(_)
        _ = Activation("tanh", name = 'final.{0}.tanh'.format(nc))(_)
        outputs = Reshape((nc*isize*isize,))(_)
        return Model(inputs=inputs, outputs=outputs)


    def build_discriminator(self, isize, nz, nc, ndf, n_extra_layers=0):
        print("discriminator:{}, {}, {}, {}, {}".format(isize, nz, nc, ndf, n_extra_layers))

        print('[INFO] Constructing discriminator model...')
        conv_init = RandomNormal(0, 0.02)
        gamma_init = RandomNormal(1., 0.02)
        assert isize%2==0
        _ = inputs = Input(shape=(1, nc*isize*isize))
        _ = Reshape((isize, isize, nc))(_)
        _ = Conv2D(filters=ndf, kernel_size=4, strides=2, use_bias=False,
                            padding = "same",
                            kernel_initializer = conv_init, 
                            name = 'initial.conv.{0}-{1}'.format(nc, ndf)             
                            ) (_)
        _ = LeakyReLU(alpha=0.2, name = 'initial.relu.{0}'.format(ndf))(_)
        csize, cndf = isize// 2, ndf
        while csize > 5:
            assert csize%2==0
            in_feat = cndf
            out_feat = cndf*2
            _ = Conv2D(filters=out_feat, kernel_size=4, strides=2, use_bias=False,
                            padding = "same",kernel_initializer = conv_init,
                            name = 'pyramid.{0}-{1}.conv'.format(in_feat, out_feat)) (_)
            if 0: # toggle batchnormalization
                _ = BatchNormalization(name = 'pyramid.{0}.batchnorm'.format(out_feat),                                   
                                    momentum=0.9, axis=-1, epsilon=1.01e-5,
                                    gamma_initializer = gamma_init)(_, training=1)        
            _ = LeakyReLU(alpha=0.2, name = 'pyramid.{0}.relu'.format(out_feat))(_)
            csize, cndf = (csize+1)//2, cndf*2
        _ = Conv2D(filters=1, kernel_size=csize, strides=1, use_bias=False,
                            kernel_initializer = conv_init,name = 'final.{0}-{1}.conv'.format(cndf, 1)) (_)
        outputs = Flatten()(_)
        return Model(inputs=inputs, outputs=outputs)




# nc = 3
# nz = 100
# ngf = 64
# ndf = 64
# n_extra_layers = 0
# Diters = 5

# image_size = 32
# batch_size = 64
# learning_rate_D = 1e-4
# learning_rate_G = 1e-4

# dcgan = DCMGAN(learning_rate_G=learning_rate_G,
#                 learning_rate_D=learning_rate_D,
#                 batch_size=batch_size,
#                 nc = nc,
#                 nz = nz,
#                 ngf = ngf,
#                 ndf = ndf,
#                 n_extra_layers = n_extra_layers,
#                 Diters = Diters,
#                 image_size = image_size)
# dcgan.discriminator.summary()
# dcgan.generator.summary()