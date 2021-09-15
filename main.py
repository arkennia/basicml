"""Simple example showcasing how to use the package."""
import mnist_loader
import numpy as np
from models import Model
import optimizers
import layers
import activations
import losses

x_train, y_train, x_test, y_test, test_data = mnist_loader.load_data_wrapper()

x_train = np.stack([x.flatten() for x in x_train])
x_test = np.stack([x.flatten() for x in x_test])

y_train = np.stack([y.flatten() for y in y_train])
y_test = np.stack([y.flatten() for y in y_test])
model = Model()
model.build(losses.CrossEntropy(apply_softmax=True), optimizers.SGD(1e-3))

model.add_layer(layers.Input())
model.add_layer(layers.Dense(30, activations.Sigmoid()))
model.add_layer(layers.Dense(10, activations.Sigmoid()))

model.fit(x_train, y_train, val_data=(x_test, y_test), epochs=10000)
