"""Loss base class."""

from abc import ABC, abstractmethod
from activations.activation import Activation
import numpy.typing as npt


class Loss(ABC):
    """
    The Loss base class.

    All classes deriving from this must implement the following functions:
        __call__
        _delta
    """

    def __init__(self) -> None:
        """Init the class."""
        super().__init__()

    @abstractmethod
    def __call__(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        """
        Output the loss or cost calculated using y_pred and y_true.

        Parameters:
        y_pred: The predicted output of a layer.
        y_true: The True output it should have given.

        Return:
        The loss of this particular training example.
        """
        pass

    @abstractmethod
    def _delta(self, z: npt.ArrayLike, y_pred: npt.ArrayLike,
               y_true: npt.ArrayLike, activation: Activation = None) -> npt.ArrayLike:
        """
        Compute the derivative of the cost function with respect to y_pred and y_true.

        Parameters:
        y_pred: The predicted output of a layer.
        y_true: The True output it should have given.

        Return:
        The derivative loss of this particular training example.
        """
        pass
