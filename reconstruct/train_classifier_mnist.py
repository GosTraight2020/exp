from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input, Reshape
import numpy as np 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype(np.float32)/255
X_test = X_test.reshape(-1, 784).astype(np.float32)/255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

inputs = Input(shape=(784))
_ = Reshape((28, 28, 1))(inputs)
_ = Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(_)
_ = MaxPooling2D(pool_size=(2, 2))(_)
_ = Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(_)
_ = MaxPooling2D(pool_size=(2, 2))(_)
_ = Dropout(0.25)(_)
_ = Flatten()(_)
_ = Dense(128, activation='relu')(_)
_ = Dropout(0.25)(_)
outputs = Dense(10, activation='softmax')(_)

model = Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=256, verbose=2)

model.save('./model/mnist_classifer.h5')