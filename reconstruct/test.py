import  tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import mean_squared_error
from utils import compute_accuracy, eucl_dist_output_shape, euclidean_distance, contrastive_loss, generate_siamese_inputs
from GAN import DCMGAN
import numpy as np
tf.enable_eager_execution()

(X_train, y_train),(X_test, y_test) = mnist.load_data()
# model = load_model("/Test/reconstruct/checkpoint/siamese_40.h5")
# # model.summary()
# X_test = X_test.astype("float32") /255.0
# X_test = X_test.reshape(-1, 784)
# class1 = X_test[y_test == 1]
# img1 = class1[1].reshape(-1, 784)
# class7 = X_test[y_test == 1]
# img2 = class7[1].reshape(-1, 784)

# v1 = model.predict(img1)
# v2 = model.predict(img2)
# test_distance = Lambda(euclidean_distance,
#                                     output_shape=eucl_dist_output_shape)([v1, v2])

# print(test_distance)
# print(y_train[0])

# y_train = to_categorical(y_train)

# print(y_train[0])

# nc = 1
# nz = 100
# ngf = 64
# ndf = 64
# n_extra_layers = 0
# Diters = 5

# image_size = 28
# batch_size = 256
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
#                 image_size = image_size,
#                 dataset='mnist',
#                 condtional=True)
# print(dcgan)

# def aprint(dcgan):
#     print(dcgan)

# aprint(dcgan)
a = np.array([ 1, 1])
b = np.array([ 2, 2])
c= mean_squared_error(a,b)
print(c)