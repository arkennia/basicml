"""Mean squared error loss."""

from losses.loss import Loss
import numpy as np
import numpy.typing as npt
from activations import Activation


class MSE(Loss):
    """Implements Mean Squared Error loss. Derives from `Loss`."""

    def __call__(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        """
        Output the loss or cost calculated using y_pred and y_true.

        Parameters:
        y_pred: The predicted output of a layer.
        y_true: The True output it should have given.

        Return:
        The loss of this particular training example.
        """
        return 0.5 * np.linalg.norm(y_pred - y_true)**2

    def _delta(self, z: npt.ArrayLike, y_pred: npt.ArrayLike,
               y_true: npt.ArrayLike, activation: Activation = None) -> npt.ArrayLike:
        """Find the derivative of the loss function with respect to y_true and y_pred."""
        if activation is not None:
            z = activation.prime(z)
        else:
            z = 1.0
        return np.transpose((y_pred - y_true) * np.transpose(z))
