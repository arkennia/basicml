"""Optimizer base class."""

from abc import ABC, abstractmethod
import numpy.typing as npt


class Optimizer(ABC):
    """
    The Optimizer base class.

    All classes that derive from this must implement:
        apply_gradients
    """

    def __init__(self, learning_rate: float) -> None:
        """Init the class and learning rate."""
        super().__init__()
        self.lr = learning_rate

    @abstractmethod
    def apply_gradients(self, gradients: npt.ArrayLike, variables: npt.ArrayLike, batch_size: int) -> npt.ArrayLike:
        """
        Apply the gradients to the variables of the model.

        Parameters:
            gradients: The gradients to adjust with.
            variables: The model's variables (weights and biases).
            batch_size: The size of each batch.

        Returns:
            The modified variables.
        """
        pass
