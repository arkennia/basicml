from activations import Activation
import numpy.typing as npt


class ReLU(Activation):
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
        if z < 0:
            return 0
        else:
            return 1

    def _apply(self, z: npt.ArrayLike) -> npt.ArrayLike:
        return max(0, z)
