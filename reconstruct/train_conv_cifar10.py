from GAN import DCMGAN
from tensorflow.keras.datasets import mnist, cifar10
from utils import generate_GAN_inputs, plot_sample_images
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 

#开启动态图模式
tf.enable_eager_execution()
tf.executing_eagerly()

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

nc = 3
nz = 100
ngf = 64
ndf = 64
n_extra_layers = 0
Diters = 5

image_size = 32
batch_size = 64
learning_rate_D = 1e-4
learning_rate_G = 1e-4

dcgan = DCMGAN(learning_rate_G=learning_rate_G,
                learning_rate_D=learning_rate_D,
                batch_size=batch_size,
                nc = nc,
                nz = nz,
                ngf = ngf,
                ndf = ndf,
                n_extra_layers = n_extra_layers,
                Diters = Diters,
                image_size = image_size,
                dataset='cifar10',
                condtional=False)
def normal_func(X, y, image_size, nc):
     X = X.reshape(-1, image_size*image_size*nc)
     X = X.astype(np.float32) /255.0
     y = y.astype(np.float32) 
     return X, y

dataset = generate_GAN_inputs(X_train, y_train, batch_size=batch_size, normal_func=normal_func, image_size=image_size, nc=nc)

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

def generate_samples(cdgan):
    z = tf.random.uniform(shape=(100, 100), minval=-1., maxval=1., dtype=tf.float32)
    return dcgan.generator.predict(z)

    
    
    
epoch_num = 200
pic_dir = './pic/conv_cifar10'
chart_dir = './chart/conv_cifar10'
D_losses = []
G_losses = []

for epoch in range(epoch_num):
    num = 0
    for((z, y), (x, eps)) in dataset:
        fake_x, loss_G= train_generator(x, y, z, eps, dcgan)
        
        for i in range(5):
            loss_D= train_discriminator(x, y, z, eps, dcgan)
        num+=1

        print("[INFO] epoch: {}, {}/{},  G_loss : {}, D_loss: {}".format(epoch, num, len(X_train)//batch_size, loss_G, loss_D))

        
    G_losses.append(loss_G)
    D_losses.append(loss_D)

    if epoch % 5 == 0:
        samples = generate_samples(dcgan)
        plot_sample_images(samples, epoch=epoch, tag='Tune', size=(-1, 32, 32, 3), dir=pic_dir)

        plt.plot(np.arange(epoch+1), G_losses)
        plt.plot(np.arange(epoch+1), D_losses)
        plt.legend()
        plt.savefig(os.path.join(chart_dir, 'standard_loss.png'))

# samples = generate_samples(dcgan)
# plot_sample_images(samples, epoch=-1, tag='Tune', size=(-1, 32, 32, 3), dir=pic_dir)
        
dcgan.generator.save("./model/standard_cifar10_generator.h5")

    


