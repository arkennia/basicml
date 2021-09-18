"""Contains a model class that implements a sequential type model."""

from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
import optimizers
import losses
import layers


class Model:
    """A sequential type model for machine learning."""

    def __init__(self):
        """Initialize the model."""
        self._layers: list[layers.Layer] = []
        self.loss: losses.Loss = None
        self.optimizer: optimizers.Optimizer = None

    def build(self, loss: losses.Loss, optimizer: optimizers.Optimizer):
        """
        Build the model with the given loss and optimizer.

        Parameters:
        loss: The type of loss to use as defined in the `losses` module.
        optimizer: The optimizer to use as defined in the `optimizers` module.

        Raises:
        ValueError: If any paramters are `None`.
        """
        if loss is not None:
            self.loss = loss
        else:
            raise ValueError("Loss cannot be none.")
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            raise ValueError("Optimizer cannot be none.")
        self._built = True

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, epochs: int = 5,
            batch_size: int = 32, val_data: Tuple[npt.ArrayLike, npt.ArrayLike] = None):
        """
        Fit the model on the data.

        Parameters:
        X: The features to train on. Should be a numpy array.
        y: The labels of the features. Should be a numpy array.
        epochs: The number of epochs to run the model for.
        batch_size: The batch_size to use. If the data does not go into `batch_size` equally,
            the remainder will be dropped.
        val_data: A tuple of (features, labels) containing the validation data.

        Raises:
        RuntimeError: If the function is called before `model.build`.
        """
        if not self._built:
            raise RuntimeError(
                "You must build the model before fitting. See: model.fit")
        x_val, y_val = val_data
        n = len(X)
        self._prepare_variables(X[0])
        for epoch in range(epochs):
            train_loss = 0.0
            val_acc = 0
            for batch in range(0, n, batch_size):
                if n - batch < batch_size:
                    continue
                train_loss = (self._train_network(
                    X[batch:batch + batch_size], y[batch:batch + batch_size]))
            v_loss, val_acc = self.evaluate(x_val, y_val)
            print(
                f"Epoch {epoch+1}:\tTrain Loss: {train_loss:.2f}\tVal Loss: {v_loss:.2f}\t\
                Val_Acc: {(val_acc/len(y_val)) * 100:.1f}%")

    def predict(self, X) -> Union[int, float]:
        """
        Predict the label of X on the model.

        Parameters:
        X: A single training example.

        Returns:
        The output of the model on `X`.

        Raises:
        RuntimeError: If the function is called before `model.build`.
        """
        if not self._built:
            raise RuntimeError(
                "You must build the model before fitting. See: model.fit")
        x = np.transpose(X)

        for layer in self._layers[1:]:
            x, _ = layer(x)

        return x

    def evaluate(self, x_test, y_test) -> Tuple[float, int]:
        """
        Evaluate the model's performance on the given data.

        Parameters:
        x_test: The features to evaluate.
        y_test: The labels to compare against.

        Return:
        (float, int): The loss on the data, and the number of correct outputs.

        Raises:
        RuntimeError: If the function is called before `model.build`.
        """
        if not self._built:
            raise RuntimeError(
                "You must build the model before fitting. See: model.fit")
        # y_pred = np.stack([np.argmax(self.predict(x)) for x in x_test])
        y_pred = np.stack([self.predict(x) for x in x_test])
        # y_test = np.stack([np.argmax(y) for y in y_test])
        loss = np.sum(self.loss(y_pred, y_test)) / len(y_test)
        y = [(np.argmax(y_p), np.argmax(y_t))
             for (y_p, y_t) in zip(y_pred, y_test)]
        # y = list(zip(y_pred, y_test))
        num_correct = sum([int(a == b) for a, b in y])
        return loss, num_correct

    def add_layer(self, layer: layers.Layer):
        """
        Add the given layer to the model's list of layers.

        Parameters:
        layer: The layer to add.

        Raises:
        ValueError: Layer is none.
        """
        if layer is None:
            raise ValueError("Layer cannot be none.")
        self._layers.append(layer)

    def _train_network(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Get the gradients by backpropagation and apply them to the weights of each layer.

        Returns:
        The loss after applying gradients.
        """
        gradient_w, gradient_b = self._backprop(x, y)

        for i, (gw, gb) in enumerate(zip(gradient_w, gradient_b)):
            # Add 1 because the gradients don't include the input layer.
            i += 1
            weights = self.optimizer.apply_gradients(
                gw, self._layers[i].get_weights(), x.shape[0])
            biases = self.optimizer.apply_gradients(
                gb, self._layers[i].get_biases(), x.shape[0])
            self._layers[i]._set_weights(weights)
            self._layers[i]._set_biases(biases)
        loss, _ = self.evaluate(x, y)
        return loss

    def _backprop(self, x, y) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        # TODO: Optimize this function.
        """
        Implement the backpropgation on a matrix consisting of `batch_size` training examples.

        The code implemented here is based on the book located
        [here](http://neuralnetworksanddeeplearning.com/index.html).
        """
        # Intialize the gradients.
        gradient_b = [np.zeros(len(b.get_biases()))
                      for b in self._layers[1:]]
        gradient_w = [np.zeros(w.get_weights().shape)
                      for w in self._layers[1:]]

        layer_activations = []
        layer_outputs = []

        # The input is given as (batch_size, training_dims)
        # However we need each trainign example as a column vector,
        # so we transpose it.
        activation = x
        layer_activations.append(activation)
        activation = np.transpose(x)

        for i in range(len(self._layers) - 1):
            # activation is the output of the layer with the activation applied.
            # layer_output is the output without the activation. Commonly refered to as `z`.
            activation, layer_output = self._layers[i + 1](activation)
            layer_activations.append(np.transpose(activation))
            layer_outputs.append(layer_output)
        activation_function = self._layers[-1].get_activation()

        # The delta requires the activation for properly calulating the delta, or change.
        delta = self.loss._delta(layer_outputs[-1],
                                 layer_activations[-1], y, activation_function)

        # Calculate gradients for final layer.
        gradient_b[-1] = np.sum(delta, axis=1)
        gradient_w[-1] = np.matmul(delta, layer_activations[-2])

        for i in range(2, len(self._layers)):
            activation_function = self._layers[-i].get_activation()

            # The first derivative of the activation function applied to the outputs.
            a_prime = activation_function.prime(layer_outputs[-i])
            weights = np.transpose(self._layers[-i + 1].get_weights())
            delta = np.matmul(weights, delta)
            delta = delta * a_prime
            gradient_b[-i] = np.sum(delta, axis=1)
            gradient_w[-i] = np.matmul(
                delta, layer_activations[-i - 1])

        return gradient_w, gradient_b

    def _prepare_variables(self, x: npt.ArrayLike):
        """Build all layers and initalize the weights and biases."""
        prev_shape = x.shape

        for i in range(1, len(self._layers)):
            self._layers[i].build(prev_shape)
            prev_shape = self._layers[i].get_output_shape()
