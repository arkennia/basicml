"""The activation base class. All other activations derives from this one."""

from abc import ABC, abstractmethod
import numpy.typing as npt


class Activation(ABC):
    """
    The activation base class.

    All activations must implement the following functions:
        __call__ : Typically will call the _apply method.
        prime: The derivative of the activation function.
        _apply: Applies this activation to the inputs.
    """

    @abstractmethod
    def __call__(self, z: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute the activation of the function on z.

        Parameters:
        z: dot(weights, inputs) + bias

        Return:
        The activation output.
        """
        pass

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
