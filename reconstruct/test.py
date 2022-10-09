import  tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import mean_squared_error
from utils import plot_sample_images, compute_accuracy, eucl_dist_output_shape, euclidean_distance, contrastive_loss, generate_siamese_inputs
from GAN import DCMGAN
import matplotlib.pyplot as plt
import numpy as np
import os
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
# a = np.array([ 1, 1])
# b = np.array([ 2, 2])
# c= mean_squared_error(a,b)
# print(c)


# def generate_conditional_sample(generator, label):
#     z = tf.random.uniform(shape=(100, 100), minval=-1., maxval=1., dtype=tf.float32)
#     y = []
#     for i in range(100):
#         y.append(label)
#     y = np.array(y, dtype=np.float32)
#     y = to_categorical(y)
#     y = y.reshape(-1, 1, 10)
#     samples = generator.predict([z, y])
#     return samples

# model = load_model("/root/exp/reconstruct/model/standard_cifar10_generator.h5")
# samples = generate_conditional_sample(model, 0)
# plot_sample_images(sample_X=samples, epoch=0, tag="Test", size=(-1, 32, 32, 3), dir='./result/mnist')


# G_loss_siamess = [-3.0337822, -2.405122, -2.2272274, -2.1595292, -2.0586724, -1.8956466, -1.7293581, -1.5877156, -1.4505074, -1.3127055, -1.179617, -1.0596203, -0.95693076, -0.87180275, -0.80200183, -0.74678546, -0.70447415, -0.67288846, -0.6484938, -0.6269847, -0.60755384, -0.5908335, -0.57584655, -0.56206363, -0.5479249, -0.5353234, -0.5234371, -0.51233333, -0.50164187, -0.49274558, -0.48340255, -0.47494227, -0.46748123, -0.45995235, -0.45261964, -0.44687796, -0.44056082, -0.43498516, -0.42970055, -0.4253283, -0.42013353, -0.4160382, -0.41094697, -0.4067313, -0.40256643, -0.39852852, -0.39442015, -0.39097485, -0.38807657, -0.38443077, -0.38173676, -0.37773556, -0.37518585, -0.3727281, -0.36814645, -0.36415955, -0.36071542, -0.35727158, -0.3543216, -0.35208777, -0.34914953, -0.346901, -0.34506974, -0.34332725, -0.34102148, -0.34010726, -0.3395512, -0.33672443, -0.33390483, -0.3326572, -0.33055526, -0.3281816, -0.32512006, -0.3230385, -0.32012773, -0.31831816, -0.31670552, -0.3153305, -0.31440872, -0.31378484, -0.3120593, -0.31058425, -0.30840084, -0.30884954, -0.30824834, -0.30563635, -0.3032007, -0.30072185, -0.29949203, -0.29814675, -0.29813665, -0.29721764, -0.2976078, -0.2972261, -0.2959743, -0.2942237, -0.29247195, -0.29086268, -0.2903492, -0.29008278]
# G_loss_mse = [-2.4105318, -2.1835139, -2.0547256, -2.038969, -1.9224032, -1.7127409, -1.570505, -1.4158919, -1.2612289, -1.1206065, -1.0019292, -0.9021655, -0.8187922, -0.7508084, -0.6963783, -0.65388083, -0.61965746, -0.5924967, -0.57047135, -0.552161, -0.53601426, -0.5223007, -0.5097139, -0.49828944, -0.48844355, -0.47978103, -0.47128004, -0.46279895, -0.45480382, -0.44750208, -0.4397449, -0.43302447, -0.42655006, -0.42098138, -0.4158585, -0.4104612, -0.40601993, -0.40137318, -0.3975851, -0.39327815, -0.3891348, -0.3856006, -0.38145226, -0.377269, -0.3741673, -0.37069294, -0.36786813, -0.36559007, -0.36495918, -0.35987994, -0.35680428, -0.35439163, -0.35163763, -0.35013026, -0.3487489, -0.34570593, -0.34195307, -0.33794093, -0.33486837, -0.33317378, -0.33167946, -0.33044845, -0.32874215, -0.33190656, -0.32793903, -0.32518864, -0.32209405, -0.32050422, -0.32124725, -0.31931666, -0.3167782, -0.3147679, -0.31348315, -0.31152794, -0.3093388, -0.30911416, -0.3088759, -0.30701166, -0.30734664, -0.3063308, -0.30300042, -0.30068797, -0.2985457, -0.29629982, -0.294999, -0.29338852, -0.2928661, -0.29368466, -0.29556566, -0.2950845, -0.29400668, -0.2923407, -0.2888543, -0.28578752, -0.28402737, -0.28237304, -0.2831463, -0.28504127, -0.2860436, -0.28418863]
# G_loss_standard = [-2.5272303, -2.2599187, -2.1803277, -2.0794744, -1.9378928, -1.7589669, -1.582489, -1.4194955, -1.2739793, -1.1528821, -1.0581777, -0.98658675, -0.93271124, -0.8916829, -0.86048824, -0.83654517, -0.8170616, -0.80165935, -0.78761476, -0.7747015, -0.762785, -0.7508714, -0.73992175, -0.7295304, -0.7202683, -0.71247536, -0.70404977, -0.69621277, -0.688843, -0.6816759, -0.6751272, -0.6686966, -0.6619889, -0.65583944, -0.64985627, -0.64469385, -0.6391457, -0.63380116, -0.6287301, -0.6239458, -0.61932033, -0.6143306, -0.60946244, -0.60461557, -0.60089755, -0.5964616, -0.5928352, -0.5885878, -0.5843552, -0.5808906, -0.57694566, -0.5738749, -0.57032293, -0.56671834, -0.5634334, -0.5602523, -0.5571784, -0.5535208, -0.55014426, -0.54757977, -0.544408, -0.5414751, -0.53860646, -0.5354983, -0.5326768, -0.5298487, -0.527196, -0.524459, -0.5218472, -0.5192527, -0.5163674, -0.51365304, -0.51139826, -0.5090391, -0.5063368, -0.5040301, -0.50153357, -0.49979818, -0.4975401, -0.49553615, -0.49315116, -0.49138528, -0.48924616, -0.48705134, -0.48483506, -0.48245937, -0.4804354, -0.4779648, -0.47630227, -0.473645, -0.47159603, -0.46977624, -0.46765912, -0.46548533, -0.46415552, -0.4619837, -0.46034828, -0.45841053, -0.4565723, -0.45465466]
# epoch = 100
# chart_dir = './chart/'
# plt.plot(np.arange(epoch), G_loss_siamess, label='siamese')
# plt.plot(np.arange(epoch), G_loss_mse, label='mse')
# plt.plot(np.arange(epoch), G_loss_standard, label='standard')
# plt.legend()
# plt.savefig(os.path.join(chart_dir, 'compare_G.png'))


from sklearn.metrics import classification_report
aux_model = load_model("./model/mnist_classifer.h5")
preds = aux_model.predict(X_train.reshape(-1, 784).astype("float32"))
y = to_categorical(y_train).astype("float32")
print(y.shape)
target = ['0', '1', '2', '3','4' ,'5','6','7','8','9']
print(y[2])
print(np.argmax(preds[2]))