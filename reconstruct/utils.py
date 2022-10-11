import tensorflow as tf
import numpy as np
import matplotlib.pyplot as  plt
import os 
from tqdm import tqdm
import random
from tensorflow.keras import backend as K
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.feature_selection import mutual_info_regression
def generate_GAN_inputs(X, y, batch_size, normal_func, image_size, nc):
    """ 输入样本X和标签y 使用函数normal对他们进行规范化
        再根据参数epoch_num和batch_size来生成数据集

    Args:
        X (np.array): 样本集合  
        y (np.array): 标签集合
        batch_size (int): 将数据集划分成batch_size大小
        normal_func (int): 对样本和标签进行规范化处理的函数

    Returns:
        tf.dataset: 处理好的数据集
    """

    X, y = normal_func(X, y, image_size, nc)
    z = tf.random.uniform(shape=(X.shape[0], 100), minval=-1., maxval=1., dtype=tf.float32)
    eps = tf.random.uniform(shape=(X.shape[0], 1), minval=0., maxval=1., dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(((z, y), (X, eps)))
    dataset = dataset.shuffle(buffer_size=X.shape[0], seed=2022)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def plot_sample_images(sample_X=None, epoch=None, tag=None, size=None, dir=None):
    sample_X = np.array(sample_X).reshape(size)
    print('[SDSDSDSD] sample_X shape:{}'.format(sample_X.shape))

    fig, axes = plt.subplots(10, 10, figsize=(50, 50))
    for i in range(10):
        for j in range(10):
            axes[i][j].imshow(sample_X[i*10+j],cmap='gray')
            axes[i][j].axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(dir, 'epoch{}_{}.png'.format(epoch, tag)))
    plt.close('all')


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1

def euclidean_distance(vec):
    x, y = vec
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy with a fixed threshold on distances.
    """
    y_true = np.squeeze(y_true, axis=1)
    pred = np.squeeze(y_pred, axis=1) < 0.5
    return np.mean(pred == y_true)
    

def generate_siamese_inputs(x, y, batch_size=None, validation=False, shape=None):
    def create_pairs(x, digit_indices):
        """ digit_indices 是数据集按类别分类的数字下标"""
        one = []
        two = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
        for d in tqdm(range(num_classes)):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                one.append(x[z1])
                two.append(x[z2])
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                one.append(x[z1])
                two.append(x[z2])
                labels.append(1)
                labels.append(0)

        one = np.array(one).reshape(shape)
        two = np.array(two).reshape(shape)
        labels = np.array(labels).reshape(-1, 1).astype(np.float32)
        return one, two, labels


    x = x.reshape(shape).astype(np.float32) / 255.0
    num_classes = np.unique(y).shape[0]
    digit_indices = [np.where(y == d)[0] for d in range(0, num_classes)]
    one, two, labels = create_pairs(x=x, digit_indices=digit_indices)

    dataset = tf.data.Dataset.from_tensor_slices(((one, two), labels))
    dataset = dataset.shuffle(buffer_size=20000, seed=2022)
    if validation:
        dataset = dataset.batch(labels.shape[0])
    else:
        dataset = dataset.batch(batch_size)
    return dataset, labels.shape[0]
    

def sigmoid(x):
    return 1. / (1 + tf.exp(-(x+2.5)))

def lrelu(x, leak, bias, g_loss=None, name="lrelu"):
    if g_loss is not None:
        bias = sigmoid(g_loss)
    x = x - bias
    with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)


def debug(num, string):
    print('[DEBUG] Point{}: {}'.format(num, string))


def cosin_distance(y_true, y_pred):
    y_true_norm = K.l2_normalize(y_true, axis=1)
    y_pred_norm = K.l2_normalize(y_pred, axis=1)
    cos_distance = K.batch_dot(y_true_norm, y_pred_norm, axes=1)
    return cos_distance

def compute_score(X_true, X_pred, label):
    scores = []
    for i in range(X_true.shape[0]):
        if label == 'mutual_info_score':
            score = mutual_info_regression(X_true[i], X_pred[i])
        elif label == 'ssim_socre':
            socre = compare_ssim(X_true[i], X_pred[i])
        else:
            raise ValueError("[ERROR] Wrong Value!")
        scores.append(score)
    scores = np.array(socres)  
    return scores