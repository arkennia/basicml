from losses.loss import Loss
import numpy as np
import numpy.typing as npt
from activations import Activation


class CrossEntropy(Loss):
    # We could create this such that it will apply softmax to the output. This is the equivalent of setting
    # from_logits=True in tensorflow. Basically we're saying in the case of logits, they aren't a probability
    # distribution.
    def __init__(self, apply_softmax=False):
        self.apply_softmax = True

    def call(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        if self.apply_softmax:
            exps = np.exp(y_pred)
            y_pred = exps / exps.sum()
        return np.sum(np.nan_to_num(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)))

    def _delta(self, z: npt.ArrayLike, y_pred: npt.ArrayLike,
               y_true: npt.ArrayLike, activation: Activation = None) -> npt.ArrayLike:
        return (y_pred - y_true)
