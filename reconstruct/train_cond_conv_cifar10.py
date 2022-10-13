from GAN import DCMGAN, New_GAN
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist, cifar10
from utils import generate_GAN_inputs, plot_sample_images, lrelu
from utils import cosin_distance, eucl_dist_output_shape, euclidean_distance, contrastive_loss
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import mean_squared_error

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 

def normal_func(X, y, image_size, nc):
     X = X.reshape(-1, image_size*image_size*nc)
     X = X.astype(np.float32) /255.0
     y = to_categorical(y)
     y = y.astype(np.float32) 
     return X, y

def train_generator(x, y, z, eps, dcgan, loss=None, aux_model=None):

    with tf.GradientTape(persistent=True) as t:
        if dcgan.condtional:
            fake_x = dcgan.generator([z, y])
        else:
            fake_x = dcgan.generator(z)
        loss_G = -tf.reduce_mean(dcgan.discriminator(fake_x)) 

        if loss == 'origin':
            aux_loss = 0
        elif loss == 'mse':
            aux_loss = mean_squared_error(x, fake_x)
            aux_loss = 10 * tf.reduce_mean(aux_loss)
        elif loss == 'siamese':
            v1 = siamese_model(fake_x)
            v2 = siamese_model(x)
            test_distance = Lambda(euclidean_distance,
                                    output_shape=eucl_dist_output_shape)([v1, v2])
            aux_loss = tf.reduce_mean(test_distance)
        elif loss == 'categorical_crossentropy':
            preds = aux_model(fake_x)
            aux_loss = K.categorical_crossentropy(y, preds)
            aux_loss = tf.reduce_mean(aux_loss)
        elif loss == 'cosin_distance':
            aux_loss = cosin_distance(x, fake_x)
            aux_loss = -tf.reduce_mean(aux_loss)
        else:
            raise ValueError("[Error] Wrong value of loss!")
        total_loss  = aux_loss + loss_G

        gradient_g = t.gradient(total_loss, dcgan.generator.trainable_variables)

    dcgan.optimizer_G.apply_gradients(zip(gradient_g, dcgan.generator.trainable_variables))
    
    return fake_x[:100], loss_G, aux_loss, loss

def train_discriminator(x, y, z, eps, dcgan):
    with tf.GradientTape(persistent=True) as t:
        if dcgan.condtional:
            fake_x = dcgan.generator([z, y])
        else:
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

def generate_conditional_sample(generator):
    z = tf.random.uniform(shape=(100, 1, 100), minval=-1., maxval=1., dtype=tf.float32)
    y = []
    for i in range(10):
        for j in range(10):
            y.append(i)
    y = np.array(y, dtype=np.float32)
    y = to_categorical(y)
    y = y.reshape(-1, 1, 10)
    samples = generator.predict([z, y])
    return samples


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
batch_size = 256
learning_rate_D = 1e-4
learning_rate_G = 1e-4
epoch_num = 100
pic_dir = './pic/conv_cifar10'
chart_dir = './chart/conv_cifar10'

dcgan = New_GAN(learning_rate_G=learning_rate_G,
                learning_rate_D=learning_rate_D,
                batch_size=batch_size,
                nc = nc,
                nz = nz,
                image_size = image_size,
                dataset = 'cifar10',
                condtional = False)

# dcgan.generator.summary()
# dcgan.discriminator.summary()

dataset = generate_GAN_inputs(X_train, y_train, batch_size=batch_size, normal_func=normal_func, image_size=image_size, nc=nc, noise_shape=100)
loss = 'origin'

if loss == 'siamese':
    aux_model = load_model("./checkpoint/siamese_cifar_100.h5")
elif loss == 'categorical_crossentropy':
    pass #TODO
else:
    aux_model = None





D_losses = []
G_losses = []
aux_losses = []
for epoch in range(epoch_num):
    num = 0
    G_temp_loss = []
    D_temp_loss = []
    aux_temp_loss = []
    for((z, y), (x, eps)) in dataset:
        fake_x, loss_G, aux_loss, loss= train_generator(x, y, z, eps, dcgan, loss, aux_model)     
        for i in range(5):
            loss_D = train_discriminator(x, y, z, eps, dcgan)
        num += 1

        # print("[INFO] epoch: {}, {}/{}, G_loss : {}, D_loss: {}".format(epoch, num, len(X_train)/batch_size,  loss_G, loss_D))
        print("[INFO] epoch: {}, {}/{}, G_loss : {}, D_loss: {}, {}: {}".format(epoch, num, len(X_train)//batch_size, loss_G, loss_D, loss, aux_loss ))
        G_temp_loss.append(loss_G)
        D_temp_loss.append(loss_D)
        aux_temp_loss.append(aux_loss)
    G_losses.append(np.mean(G_temp_loss))
    D_losses.append(np.mean(D_temp_loss))
    aux_losses.append(np.mean(aux_temp_loss))

    if epoch % 5 == 0:
        if dcgan.condtional:
            cond_samples = generate_conditional_sample(dcgan.generator)
        else:
            cond_samples = fake_x
        plot_sample_images(cond_samples, epoch=epoch, tag='{}_cifar10'.format(loss), size=(-1, image_size, image_size, nc), dir=pic_dir)
        plt.plot(np.arange(epoch+1), G_losses, label='G_loss')
        plt.plot(np.arange(epoch+1), D_losses, label='D_loss')
        plt.plot(np.arange(epoch+1), aux_losses, label=loss)
        plt.legend()
        plt.savefig(os.path.join(chart_dir, '{}_cifar10.png'.format(loss)))
        
dcgan.generator.save('./checkpoint/cifar10_generator_cond.h5')

file1 = open("./data/mnist_{}_cifar10.log".format(loss), 'w')
file1.write("G_loss: \n")
file1.write(str(G_losses))
file1.write('\n')
file1.write("D_loss:\n")
file1.write(str(D_losses))
file1.write("aux_loss:\n")
file1.write(str(aux_losses))
file1.close()


