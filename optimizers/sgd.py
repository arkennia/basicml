
from optimizers.optimizer import Optimizer
import numpy.typing as npt


class SGD(Optimizer):
    def __init__(self, learning_rate) -> None:
        super().__init__(learning_rate)

    def apply_gradients(self, gradients: npt.ArrayLike, variables: npt.ArrayLike):
        variables = [w - (self.lr) * gw
                     for w, gw in zip(variables, gradients)]
