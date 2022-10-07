from GAN import DCMGAN
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist, cifar10
from utils import generate_GAN_inputs, plot_sample_images
from utils import eucl_dist_output_shape, euclidean_distance, contrastive_loss
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 

#开启动态图模式
tf.enable_eager_execution()
tf.executing_eagerly()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

nc = 1
nz = 100
ngf = 64
ndf = 64
n_extra_layers = 0
Diters = 5

image_size = 28
batch_size = 256
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
                dataset='mnist',
                condtional=True)

# print("[DEBUG] The model of generator")
# dcgan.generator.summary()
# print("[DEBUG] The model of discriminator")
# dcgan.discriminator.summary()

def normal_func(X, y, image_size, nc):
     X = X.reshape(-1, image_size*image_size*nc)
     X = X.astype(np.float32) /255.0
     y = to_categorical(y)
     y = y.astype(np.float32) 
     return X, y

dataset = generate_GAN_inputs(X_train, y_train, batch_size=128, normal_func=normal_func, image_size=image_size, nc=nc)

def train_generator(x, y, z, eps, dcgan, siamese_model):

    with tf.GradientTape(persistent=True) as t:
        fake_x = dcgan.generator([z, y])
        loss_G = -tf.reduce_mean(dcgan.discriminator(fake_x)) 

        # =========================standard=======================================
        # gradient_g = t.gradient(loss_G, dcgan.generator.trainable_variables)
        # =========================MSE LOSS=======================================
        # mse_loss = tf.reduce_sum(tf.square(tf.subtract(x, fake_x)))/28*28/6000
        # =========================siamese LOSS====================================
        v1 = siamese_model.predict(fake_x)
        v2 = siamese_model.predict(x)
        test_distance = Lambda(euclidean_distance,
                                    output_shape=eucl_dist_output_shape)([v1, v2])

        siamese_loss = tf.reduce_mean(test_distance)

        total_loss  = siamese_loss + loss_G

        gradient_g = t.gradient(total_loss, dcgan.generator.trainable_variables)

    dcgan.optimizer_G.apply_gradients(zip(gradient_g, dcgan.generator.trainable_variables))
    
    axu_loss = siamese_loss

    return fake_x[:100], loss_G, axu_loss

def train_discriminator(x, y, z, eps, dcgan):
    with tf.GradientTape(persistent=True) as t:
        fake_x = dcgan.generator([z, y])
        x_inter = eps*x + (1-eps)*fake_x
        temp_x = dcgan.discriminator(x_inter)
        grad = t.gradient(temp_x, x_inter)
        grad_norm = tf.sqrt(tf.reduce_sum(grad**2, axis=1))
        grad_pen = 10* tf.reduce_mean(tf.nn.relu(grad_norm-1.))

        loss_D = tf.reduce_mean(dcgan.discriminator(fake_x)) - tf.reduce_mean(dcgan.discriminator(x)) + grad_pen
       

        gradient_d = t.gradient(loss_D, dcgan.discriminator.trainable_variables)

    dcgan.optimizer_D.apply_gradients(zip(gradient_d, dcgan.discriminator.trainable_variables))

    return loss_D

def generate_conditional_sample(generator):
    z = tf.random.uniform(shape=(100, 1, 100), minval=-1., maxval=1., dtype=tf.float32)
    y = []
    for i in range(10):
        for j in range(10):
            y.append(i)
    y = np.array(y, dtype=np.float32)
    y = to_categorical(y)
    samples = generator.predict([z, y])
    return samples

epoch_num = 100
pic_dir = '/Test/reconstruct/pic/conv_mnist'
chart_dir = '/Test/reconstruct/chart/conv_mnist'
D_losses = []
G_losses = []
siamese_model = load_model("/Test/reconstruct/checkpoint/siamese_40.h5")


dcgan.generator.summary()

for epoch in range(epoch_num):
    num = 0
    for((z, y), (x, eps)) in dataset:

        fake_x, loss_G, mse_loss= train_generator(x, y, z, eps, dcgan, siamese_model)        
        for i in range(5):
            loss_D = train_discriminator(x, y, z, eps, dcgan)
        num += 1
        # print("[INFO] epoch: {}, {}/{}, G_loss : {}, D_loss: {}".format(epoch, num, len(X_train)/batch_size,  loss_G, loss_D))
        print("[INFO] epoch: {}, {}/{}, G_loss : {}, D_loss: {}, mse_loss: {}".format(epoch, num, len(X_train)//128, loss_G, loss_D, mse_loss))

    G_losses.append(loss_G)
    D_losses.append(loss_D)

    if epoch % 5 == 0:
        cond_samples = generate_conditional_sample(dcgan.generator)
        plot_sample_images(cond_samples, epoch=epoch, tag='Siamese', size=(-1, image_size, image_size, nc), dir=pic_dir)

        plt.plot(np.arange(epoch+1), G_losses)
        plt.plot(np.arange(epoch+1), D_losses)
        plt.legend()
        plt.savefig(os.path.join(chart_dir, 'conditional_Siamese_loss.png'))

    


