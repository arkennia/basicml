"""Rectified Linear Unit activaiton."""

from activations import Activation
import numpy.typing as npt


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
        return self.apply(z)

    def prime(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute the output of the derivative of ReLU on z.

        Parameters:
        z: dot(weights, inputs) + bias

        Return:
        The output of the derivative of ReLU.
        """
        if z < 0:
            return 0
        else:
            return 1

    def _apply(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """Compute ReLU on z."""
        return max(0, z)
