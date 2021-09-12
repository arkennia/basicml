from activations.activation import Activation
import numpy as np
import numpy.typing as npt


class Softmax(Activation):
    def call(self, z: npt.ArrayLike) -> npt.ArrayLike:
        return self._apply(z)

    def prime(self, z: npt.ArrayLike) -> npt.ArrayLike:
        #  z = np.dot(weight, activation) + bias
        z = self._apply(z)
        z = np.reshape(z, (1, -1))
        # Diagflat is the same as z * np.identity(z)
        z = np.diagflat(z) - np.dot(z, np.transpose(z))
        return z

    def _apply(self, z: npt.ArrayLike) -> npt.ArrayLike:
        exps = np.exp(z)
        z = exps / exps.sum()
        return z
