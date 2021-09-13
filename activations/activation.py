from abc import ABC, abstractmethod
import numpy.typing as npt


class Activation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def prime(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute the output of the derivative of the activation function on z.

        Parameters:
        z: dot(weights, inputs) + bias

        Return:
        The output of the derivative of the activation function given z.
        """
        pass

    @abstractmethod
    def _apply(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute the activation of the function on z.

        Parameters:
        z: dot(weights, inputs) + bias

        Return:
        The activation output.
        """
        pass
