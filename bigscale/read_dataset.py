import os
from PIL import Image
from PIL import Image as Img
from tqdm import tqdm
import numpy as np
import h5py



data_dir = '/celeA/img_align_celeba/img_align_celeba/'

X_train = []
ORIG_WIDTH = 178
ORIG_HEIGHT = 208
diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2

WIDTH = 128
HEIGHT = 128

crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)

for img in tqdm(os.listdir(data_dir)):
    pic = Image.open(data_dir + img).crop(crop_rect)
    pic.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
    X_train.append(np.uint8(pic))
    
X_train = np.array(X_train).astype(np.float32)/255.0
print(X_train.shape)

hfile = h5py.File('./dataset/celeA.h5', 'w')
hfile.create_dataset("X_train", data=X_train)
hfile.close()
    
    