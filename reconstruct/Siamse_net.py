from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Lambda, Input
from utils import debug, compute_accuracy, eucl_dist_output_shape, euclidean_distance, contrastive_loss, generate_siamese_inputs
from tensorflow.keras.datasets import mnist, fashion_mnist
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

class Siamese_Net:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(784,), activation='relu'))
        # model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        return model

    def train_one_step(self, x1, x2, y):
        with tf.GradientTape() as t:
            vector_a = self.model(x1)
            vector_b = self.model(x2)
            distance = Lambda(euclidean_distance,
                                output_shape=eucl_dist_output_shape)([vector_a, vector_b])
            loss = contrastive_loss(y_true=y, y_pred=distance)
            
            gradients = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

tf.enable_eager_execution()
epoch_num = 40
batch_size = 128
siamese_net = Siamese_Net(learning_rate=1e-4)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
train_dataset, data_size = generate_siamese_inputs(X_train, y_train, batch_size, shape=(-1, 784))
val_dataset, _ = generate_siamese_inputs(X_train, y_train, validation=True, shape=(-1, 784))
test_dataset, _ = generate_siamese_inputs(X_test, y_test, validation=True, shape=(-1, 784))



for epoch in range(epoch_num):
    num = 0
    for ((x1, x2), y) in train_dataset:
        loss = siamese_net.train_one_step(x1, x2, y)
        print("[INFO]epoch:{}, {}/{}, loss = {}".format(epoch, num, data_size//batch_size, loss))
        num+=1

for((val_X_1, val_X_2), val_label) in val_dataset:
    process_a = siamese_net.model.predict(val_X_1)
    process_b = siamese_net.model.predict(val_X_2)
    val_distance = Lambda(euclidean_distance,
                                    output_shape=eucl_dist_output_shape)([process_a, process_b])

    val_acc = compute_accuracy(y_true=val_label, y_pred=val_distance)
    print('[RESULT] Acc of validation dataset is {}'.format(val_acc))



for((test_X_1, test_X_2), test_label) in test_dataset:
    process_a = siamese_net.model.predict(test_X_1)
    process_b = siamese_net.model.predict(test_X_2)
    test_distance = Lambda(euclidean_distance,
                                    output_shape=eucl_dist_output_shape)([process_a, process_b])

    test_acc = compute_accuracy(y_true=test_label, y_pred=test_distance)
    print('[RESULT] Acc of test dataset is {}'.format(test_acc))

siamese_net.model.summary()
siamese_net.model.save("./checkpoint/siamese_fashion_mnist_40.h5")

