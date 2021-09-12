from losses.loss import Loss
import numpy as np
import numpy.typing as npt
from activations.activation import Activation


class MSE(Loss):
    def call(self, y_pred, y_true):
        return 0.5 * np.linalg.norm(y_pred - y_true)**2

    def _delta(self, z: npt.ArrayLike, y_pred: npt.ArrayLike,
               y_true: npt.ArrayLike, activation: Activation = None) -> npt.ArrayLike:
        if activation is not None:
            z = activation.prime(z)
        else:
            z = 1.0
        return (y_pred - y_true) * z
