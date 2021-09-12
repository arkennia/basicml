from abc import ABC, abstractmethod
import numpy.typing as npt


class Optimizer(ABC):
    def __init__(self, learning_rate: float) -> None:
        super().__init__()
        self.lr = learning_rate

    @abstractmethod
    def apply_gradients(self, gradients: npt.ArrayLike, variables: npt.ArrayLike):
        """
        Apply the gradients to the variables of the model.

        Parameters:
        gradients: The gradients to adjust with.
        variables: The model's variables (weights and biases).
        """
        pass
