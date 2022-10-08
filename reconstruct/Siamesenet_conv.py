from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, Input, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, Flatten, Lambda
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from utils import compute_accuracy, eucl_dist_output_shape, euclidean_distance, contrastive_loss, generate_siamese_inputs, debug
import tensorflow as tf
import numpy.random as rng
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow.keras import backend as K
from tqdm import tqdm
import random



tf.enable_eager_execution()

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def create_pairs(x, digit_indices):
    
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]] # positive sample
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i] # negative sample
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def fire_module(x, s1x1, e1x1, e3x3, name):
    #Squeeze layer
    squeeze = Conv2D(s1x1, (1, 1), activation='relu', padding='valid', kernel_initializer='glorot_uniform', name = name + 's1x1')(x)
    squeeze_bn = BatchNormalization(name=name+'sbn')(squeeze)
    
    #Expand 1x1 layer and 3x3 layer are parallel

    #Expand 1x1 layer
    expand1x1 = Conv2D(e1x1, (1, 1), activation='relu', padding='valid', kernel_initializer='glorot_uniform', name = name + 'e1x1')(squeeze_bn)
    
    #Expand 3x3 layer
    expand3x3 = Conv2D(e3x3, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform', name = name +  'e3x3')(squeeze_bn)
    
    #Concatenate expand1x1 and expand 3x3 at filters
    output = Concatenate(axis = 3, name=name)([expand1x1, expand3x3])
    
    return output
  
    

def W_init(shape,dtype=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,dtype=dtype)

def b_init(shape,dtype=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,dtype=dtype)



class Siamese_Net:
    def __init__(self, learning_rate, input_shape):
        self.learning_rate = learning_rate
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        conv1 = Conv2D(96, kernel_size=(3, 3), strides=(2, 2),  padding='same', activation='relu', name = 'Conv1')(inputs)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Maxpool1')(conv1)
        batch1 = BatchNormalization(name='Batch1')(maxpool1)
    #     fire2 = fire_module(batch1, 16, 64, 64, "Fire2")
    #     fire3 = fire_module(fire2, 16, 64, 64, "Fire3")
        fire4 = fire_module(batch1, 32, 128, 128, "Fire2")
        maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Maxpool2')(fire4)
    #     fire5 = fire_module(maxpool4, 32, 128, 128, "Fire5")
        fire6 = fire_module(maxpool4, 48, 192, 192, "Fire3")
        fire7 = fire_module(fire6, 48, 192, 192, "Fire4")
        fire8 = fire_module(fire7, 48, 192, 192, "Fire5")
        maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Maxpool5')(fire8)
    #     fire9 = fire_module(maxpool8, 64, 256, 256, "Fire9")
        dropout = Dropout(0.5, name="Dropout")(maxpool8)
        conv10 = Conv2D(10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='Conv6')(dropout)
        batch10 = BatchNormalization(name='Batch6')(conv10)
        avgpool10 = GlobalAveragePooling2D(name='GlobalAvgPool6')(batch10)
        #softmax = Activation('softmax')(avgpool10)
        
        squeezenet = Model(inputs=inputs, outputs=avgpool10)
        return squeezenet
            # return model
            


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

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
siamese_net = Siamese_Net(learning_rate = 1e-3, input_shape=(32, 32, 3))
siamese_net.model.summary()

batch_size = 128
epoch_num = 100
num_classes = np.unique(y_train).shape[0]
digit_indices_train = [np.where(y_train == d)[0] for d in range(0, num_classes)]
digit_indices_test = [np.where(y_test == d)[0] for d in range(0, num_classes)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices_train)
te_pairs, te_y = create_pairs(X_test, digit_indices_test)

input_a = Input(shape=siamese_net.input_shape, name='input_a')
input_b = Input(shape=siamese_net.input_shape, name='input_b')

processed_a = siamese_net.model(input_a)
processed_b = siamese_net.model(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)
model.compile(loss=contrastive_loss, optimizer=siamese_net.optimizer, metrics=[accuracy])

history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epoch_num,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
siamese_net.model.save("/paper/reconstruct/checkpoint/siamese_cifar_100.h5")