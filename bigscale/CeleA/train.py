from DCWGAN import DCWGAN
from tensorflow.keras.datasets import mnist, cifar10
from utils import generate_GAN_inputs, plot_sample_images
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 
import h5py

#开启动态图模式
tf.enable_eager_execution()
tf.executing_eagerly()

hfile = h5py.File('../dataset/celeA.h5', 'r')
X_train = hfile['X_train']
X_train = np.array(X_train)


image_size = 128
batch_size = 16
learning_rate_D = 1e-4
learning_rate_G = 1e-4

dcgan = DCWGAN(learning_rate_G=learning_rate_G,
                learning_rate_D=learning_rate_D,
                image_size = image_size)

def normal_func(X, y):
     X = X.reshape(-1, 128*128*3)
     y = y.astype(np.float32) 
     return X, y

dataset = generate_GAN_inputs(X_train, np.zeros(X_train.shape), batch_size=batch_size, normal_func=normal_func)

def train_generator(x, y, z, eps, dcgan):

    with tf.GradientTape(persistent=True) as t:
        fake_x = dcgan.generator(z)
        loss_G = -tf.reduce_mean(dcgan.discriminator(fake_x))
        gradient_g = t.gradient(loss_G, dcgan.generator.trainable_variables)

    dcgan.optimizer_G.apply_gradients(zip(gradient_g, dcgan.generator.trainable_variables))

    return fake_x[:100], loss_G

def train_discriminator(x, y, z, eps, dcgan):
    with tf.GradientTape(persistent=True) as t:
        fake_x = dcgan.generator(z)
        x_inter = eps*x + (1-eps)*fake_x
        temp_x = dcgan.discriminator(x_inter)
        grad = t.gradient(temp_x, x_inter)
        grad_norm = tf.sqrt(tf.reduce_sum(grad**2, axis=1))
        grad_pen = 10* tf.reduce_mean(tf.nn.relu(grad_norm-1.))

        loss_D = tf.reduce_mean(dcgan.discriminator(fake_x)) - tf.reduce_mean(dcgan.discriminator(x)) + grad_pen
        gradient_d = t.gradient(loss_D, dcgan.discriminator.trainable_variables)

    dcgan.optimizer_D.apply_gradients(zip(gradient_d, dcgan.discriminator.trainable_variables))

    return loss_D


epoch_num = 100
pic_dir = './pic/'
chart_dir = './chart/'
D_losses = []
G_losses = []

for epoch in range(epoch_num):
    for((z, y), (x, eps)) in dataset:
        fake_x, loss_G= train_generator(x, y, z, eps, dcgan)
        
        for i in range(5):
            loss_D= train_discriminator(x, y, z, eps, dcgan)

        print("[INFO] epoch: {}, G_loss : {}, D_loss: {}".format(epoch, loss_G, loss_D))

        
    G_losses.append(loss_G)
    D_losses.append(loss_D)

    if epoch % 10 == 0:
        print(fake_x)
        plot_sample_images(fake_x, epoch=epoch, tag='Tune', size=(-1, 128, 128, 3), dir=pic_dir)

        plt.plot(np.arange(epoch+1), G_losses)
        plt.plot(np.arange(epoch+1), D_losses)
        plt.legend()
        plt.savefig(os.path.join(chart_dir, 'loss.png'))

    


