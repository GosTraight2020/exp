from skimage.segmentation import slic, mark_boundaries
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, MaxPooling2D
from utils import plot_sample_images
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import h5py

def build_encoder(input_shape):
    inputs = Input(shape=input_shape)
    outputs = MaxPooling2D()(inputs)
    return Model(inputs, outputs)

def mean_slic(images, n_segments, compactness, sigma, gray_scale=True):
    images = np.array(images).reshape(-1, 28, 28, 1)
    result = []
    for image in tqdm(images):
        segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma)
        segmentation_num = np.max(segments)
        if gray_scale:
            for i in range(1, segmentation_num+1):
                val = np.mean(image[segments == i])
                image[segments == i] = val
        else:
            r, g, b = cv2.split(image)
            for c in [r, g, b]:
                for i in range(1, segmentation_num+1):
                    mean = np.mean(c[segments == i])
                    c[segments == i] = mean
            image = cv2.merget([r, g, b])        
        result.append(image)
    result = np.array(result)
    return result

def encode(images, n_segments, compactness, sigma, gray_scale, input_shape):
    _ = mean_slic(images, n_segments, compactness, sigma, gray_scale)
    print(_.shape)
    model = build_encoder(input_shape)
    processed = model.predict(_)
    return processed
    
    
    
(X_train, y_train),(X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype(np.float32)/255
result = encode(X_train, n_segments=20, compactness=1, sigma=0, gray_scale=True, input_shape=(28, 28, 1))
plot_sample_images(sample_X=result[:100], epoch=0, tag="TEST", size=(-1, 14, 14, 1), dir="./pic")

hfile = h5py.File("./data/n_segments_20_compactnes_1_sigma_1.h5", 'w')
hfile.create_dataset("feature", data=result)
hfile.create_dataset("label", data=y_train)
hfile.close()