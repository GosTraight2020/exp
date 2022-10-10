from GAN import DCMGAN
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.losses import categorical_crossentropy
from utils import generate_GAN_inputs, plot_sample_images, lrelu, debug
from utils import eucl_dist_output_shape, euclidean_distance, contrastive_loss, cosin_distance
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 


def train_generator(x, y, z, eps, dcgan, loss=None, siamese_model=None, template_samples=None):
 
    # 如果参数templates_samples不为空，说明需要按照输入的模版样本来引导样本生成
    if template_samples is not None:
        templates = x
        y_templates = np.argmax(y, axis=1)
        for i in range(10):
            templates[y_templates==i] = template_samples[i]
    else:
        templates = x
            
            
    with tf.GradientTape(persistent=True) as t:
        fake_x = dcgan.generator([z, y])
        loss_G = -tf.reduce_mean(dcgan.discriminator(fake_x)) 
        
        if loss == 'origin':
            aux_loss = 0
        elif loss == 'mse':
            aux_loss = mean_squared_error(templates, fake_x)
            aux_loss = 10*tf.reduce_mean(axu_loss)
        elif loss == 'siamese':
            v1 = siamese_model(fake_x)
            v2 = siamese_model(templates)
            test_distance = Lambda(euclidean_distance,
                                    output_shape=eucl_dist_output_shape)([v1, v2])
            aux_loss = tf.reduce_mean(test_distance)
        elif loss == 'categorical_crossentropy':
            preds = aux_model(fake_x)
            aux_loss = K.categorical_crossentropy(y, preds)
            aux_loss = tf.reduce_mean(aux_loss)
        elif loss == 'cosin_distance':
            aux_loss = cosin_distance(templates, fake_x)
            aux_loss = -tf.reduce_mean(aux_loss)
        else:
            raise ValueError("[Error] Wrong value of loss!")

        total_loss  = aux_loss + loss_G

        gradient_g = t.gradient(total_loss, dcgan.generator.trainable_variables)

    dcgan.optimizer_G.apply_gradients(zip(gradient_g, dcgan.generator.trainable_variables))
    

    return fake_x[:100], loss_G, aux_loss, loss

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
    y = y.reshape(-1, 1, 10)
    samples = generator.predict([z, y])
    return samples

def generate_templates(X_train, y_train):
    """generate template samples from X_train, one for each category

    Args:
        X_train (nD_array): training samples
        y_train (nD_array): training labels

    Returns:
        1D_array: one templates sample for each category
    """
    templates = []
    for i in range(9):
        template = X_train[y_train == i][0]
        templates.append(template)
    templates = np.array(templates)
    return templates


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

epoch_num = 100
loss = 'cosin_distance'
enable_template = True

pic_dir = './pic/conv_mnist'
chart_dir = './chart/conv_mnist'


if loss == 'siamese':
    aux_model = load_model("./checkpoint/siamese_40.h5")
elif loss == 'categorical_crossentropy':
    aux_model = load_model("./model/mnist_classifer.h5")
else:
    aux_model = None

if enable_template = True:
    template_samples = generate_templates(X_train, y_train)
else:
    template_samples = None

def normal_func(X, y, image_size, nc):
     X = X.reshape(-1, image_size*image_size*nc)
     X = X.astype(np.float32) /255.0
     y = to_categorical(y)
     y = y.astype(np.float32) 
     return X, y

dataset = generate_GAN_inputs(X_train, y_train, batch_size=batch_size, normal_func=normal_func, image_size=image_size, nc=nc)

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




D_losses = []
G_losses = []
for epoch in range(epoch_num):
    num = 0
    G_temp_loss = []
    D_temp_loss = []
    for((z, y), (x, eps)) in dataset:
        fake_x, loss_G, axu_loss, loss= train_generator(x, y, z, eps, dcgan, loss, aux_model, template_samples)     
        for i in range(5):
            loss_D = train_discriminator(x, y, z, eps, dcgan)
        num += 1
        # print("[INFO] epoch: {}, {}/{}, G_loss : {}, D_loss: {}".format(epoch, num, len(X_train)/batch_size,  loss_G, loss_D))
        print("[INFO] epoch: {}, {}/{}, G_loss : {}, D_loss: {}, {}_loss: {}".format(epoch, num, len(X_train)//batch_size, loss_G, loss_D, loss, axu_loss))
        G_temp_loss.append(loss_G)
        D_temp_loss.append(loss_D)
    G_losses.append(np.mean(G_temp_loss))
    D_losses.append(np.mean(D_temp_loss))

    if epoch % 5 == 0:
        cond_samples = generate_conditional_sample(dcgan.generator)
        plot_sample_images(cond_samples, epoch=epoch, tag=loss, size=(-1, image_size, image_size, nc), dir=pic_dir)

        plt.plot(np.arange(epoch+1), G_losses)
        plt.plot(np.arange(epoch+1), D_losses)
        plt.legend()
        plt.savefig(os.path.join(chart_dir, 'conditional_{}_loss_batch_256_template{}.png'.format(loss, enable_template)))

    
file1 = open("./data/mnist_{}_template_{}.log".format(loss, enable_template), 'w')
file1.write("G_loss: \n")
file1.write(str(G_losses))
file1.write('\n')
file1.write("D_loss:\n")
file1.write(str(D_losses))
file1.close()

dcgan.generator.save("./model/mnist_{}_template_{}.h5".format(loss, enable_template))