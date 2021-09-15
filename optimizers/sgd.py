"""Stochastic Gradient Descent optimizer."""


from optimizers.optimizer import Optimizer
import numpy.typing as npt
import numpy as np


class SGD(Optimizer):
    """Implements SGD. Derives from `Optimizer`."""

    def __init__(self, learning_rate) -> None:
        """Init the class and learning rate."""
        super().__init__(learning_rate)

    def apply_gradients(self, gradients: npt.ArrayLike, variables: npt.ArrayLike) -> npt.ArrayLike:
        """
        Apply the gradients to the variables of the model.

        Parameters:
            gradients: The gradients to adjust with.
            variables: The model's variables (weights and biases).

        Returns:
            The modified variables.
        """
        variables = [w - (self.lr) * gw
                     for w, gw in zip(variables, gradients)]
        return np.array(variables)
