"""Simple example showcasing how to use the package."""
import numpy as np
from models import Model
import optimizers
import layers
import activations
import losses
from keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

# Normalize inputs
x_train = x_train / 255
x_test = x_test / 255

encoder = LabelBinarizer()
encoder.fit(np.append(y_train, y_test))

y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

model = Model()
model.build(losses.CrossEntropy(apply_softmax=True), optimizers.SGD(1e-3))

model.add_layer(layers.Input())
# model.add_layer(layers.Dense(60, activations.ReLU()))
model.add_layer(layers.Dense(30, activations.ReLU()))
model.add_layer(layers.Dense(10))

model.fit(x_train, y_train, val_data=(x_test, y_test), epochs=15)
