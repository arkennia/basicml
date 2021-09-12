from activations.activation import Activation
import numpy.typing as npt


class ReLU(Activation):
    def call(self, z: npt.ArrayLike) -> npt.ArrayLike:
        return self.apply(z)

    def prime(self, z: npt.ArrayLike) -> npt.ArrayLike:
        if z < 0:
            return 0
        else:
            return 1

    def _apply(self, z: npt.ArrayLike) -> npt.ArrayLike:
        return max(0, z)
