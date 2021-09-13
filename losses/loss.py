from abc import ABC, abstractmethod
from activations.activation import Activation
import numpy.typing as npt


class Loss(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _delta(self, z: npt.ArrayLike, y_pred: npt.ArrayLike,
               y_true: npt.ArrayLike, activation: Activation = None) -> npt.ArrayLike:
        """
        Compute the derivative of the cost function with respect to y_pred and y_true.
        In the case of CrossEntropy, that is just (y_pred - y_true).

        Parameters:
        y_pred: The predicted output of a layer.
        y_true: The True output it should have given.

        Return:
        The derivative loss of this particular training example.
        """
        pass
