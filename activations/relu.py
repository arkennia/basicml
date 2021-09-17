"""Rectified Linear Unit activaiton."""

from activations import Activation
import numpy.typing as npt
import numpy as np


class ReLU(Activation):
    """Implements Refticifed Linear Unit activation. Derives from `Activation`."""

    def __call__(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute the activation of the function on z.

        Parameters:
        z: dot(weights, inputs) + bias

        Return:
        The activation output.
        """
        return self._apply(z)

    def prime(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute the output of the derivative of ReLU on z.

        Parameters:
        z: dot(weights, inputs) + bias

        Return:
        The output of the derivative of ReLU.
        """
        return np.vectorize(lambda x: 1 if x > 0 else 0)(z)

    def _apply(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """Compute ReLU on z."""
        return np.vectorize(lambda x: max(0, x))(z)
