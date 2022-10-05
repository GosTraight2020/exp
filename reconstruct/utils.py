import tensorflow as tf
import numpy as np
import matplotlib.pyplot as  plt
import os 
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
            axes[i][j].imshow(sample_X[i*10+j])
            axes[i][j].axis('off')
    plt.tight_layout()

    plt.savefig(os.path.join(dir, 'epoch{}_{}.png'.format(epoch, tag)))
    plt.close('all')
    

    
