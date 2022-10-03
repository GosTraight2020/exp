from GAN import MGAN
from tensorflow.keras.datasets import mnist,cifar10
from utils import generate_GAN_inputs, plot_sample_images
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 

#开启动态图模式
tf.enable_eager_execution()
tf.executing_eagerly()

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

model_shape_G = [100, 32, 128, 256, 512, 1024, 2048, 3072]
model_shape_D = [3072, 2048, 1024, 512, 256, 128, 32, 1]
learning_rate_G = 1e-4
learning_rate_D = 1e-4
batch_size = 128

mgan = MGAN(model_shape_G, model_shape_D, learning_rate_G ,learning_rate_D, batch_size, X_train, y_train, X_test, y_test)

# def normal_func(X, y):
#      X = X.reshape(-1, 784)
#      X = X.astype(np.float32) /255.0
#      y = y.astype(np.float32) 
#      return X, y

def normal_func(X, y):
     X = X.reshape(-1, 3072)
     X = X.astype(np.float32) /255.0
     y = y.astype(np.float32) 
     return X, y

dataset = generate_GAN_inputs(X_train, y_train, batch_size=128, epoch_num=100, normal_func=normal_func)

def train_generator(x, y, z, eps, mgan):

    with tf.GradientTape(persistent=True) as t:
        fake_x = mgan.generator(z)
        loss_G = -tf.reduce_mean(mgan.discriminator(fake_x))
        gradient_g = t.gradient(loss_G, mgan.generator.trainable_variables)

    mgan.optimizer_G.apply_gradients(zip(gradient_g, mgan.generator.trainable_variables))

    return fake_x[:100], loss_G

def train_discriminator(x, y, z, eps, mgan):

    with tf.GradientTape(persistent=True) as t:
        fake_x = mgan.generator(z)
        X_inter = eps*x + (1-eps)*fake_x
        temp_x = mgan.discriminator(X_inter)
        grad = t.gradient(temp_x, X_inter)
        grad_norm = tf.sqrt(tf.reduce_sum(grad**2, axis=1))
        grad_pen = 10* tf.reduce_mean(tf.nn.relu(grad_norm-1.))

        loss_D = tf.reduce_mean(mgan.discriminator(fake_x)) - tf.reduce_mean(mgan.discriminator(x)) + grad_pen
        gradient_d = t.gradient(loss_D, mgan.discriminator.trainable_variables)

    mgan.optimizer_D.apply_gradients(zip(gradient_d, mgan.discriminator.trainable_variables))

    return loss_D


epoch_num = 100
pic_dir = '/Test/reconstruct/pic'
chart_dir = '/Test/reconstruct/chart'
D_losses = []
G_losses = []

for epoch in range(epoch_num):
    for((z, y), (x, eps)) in dataset:
        fake_x, loss_G= train_generator(x, y, z, eps, mgan)
        for i in range(5):
            loss_D= train_discriminator(x, y, z, eps, mgan)
        print("[INFO] epoch: {}, G_loss : {}, D_loss: {}".format(epoch, loss_G, loss_D))

    G_losses.append(loss_G)
    D_losses.append(loss_D)

    if epoch % 10 == 0:
        plot_sample_images(fake_x, epoch=epoch, tag='Tune', size=(-1, 32, 32, 3), dir=pic_dir)

        plt.plot(np.arange(epoch+1), G_losses)
        plt.plot(np.arange(epoch+1), D_losses)
        plt.legend()
        plt.savefig(os.path.join(chart_dir, 'loss.png'))

    


