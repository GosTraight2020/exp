from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

tf.enable_eager_execution()

def cosin_distance(y_true, y_pred):
    y_true_norm = K.l2_normalize(y_true, axis=1)
    print(y_true_norm)
    y_pred_norm = K.l2_normalize(y_pred, axis=1)
    print(y_pred_norm)
    cos_distance = K.batch_dot(y_true_norm, y_pred_norm, axes=1)
    print(cos_distance)
    return cos_distance

(X_train, y_train), (_) = mnist.load_data()
X_train  =  X_train.reshape(-1, 784).astype(np.float32)/255.0

class1 = X_train[y_train==1][0].reshape(-1, 784)
class7 = X_train[y_train==7][0].reshape(-1, 784)
class2 = X_train[y_train==2][0]

a = np.array([[1,2,3], [1, 2, 3]], dtype=np.float32)
b = np.array([[1,2,3], [1, 1, 1]], dtype=np.float32)

cosin = cosin_distance(a, b)
print('图片余弦相似度', cosin)
