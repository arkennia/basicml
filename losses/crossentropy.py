"""CrossEntropy loss implementation."""

from losses.loss import Loss
import numpy as np
import numpy.typing as npt
from activations import Activation


class CrossEntropy(Loss):
    """Implements loss for cross entropy with and without a softmax."""

    # We could create this such that it will apply softmax to the output. This is the equivalent of setting
    # from_logits=True in tensorflow. Basically we're saying in the case of logits, they aren't a probability
    # distribution.

    def __init__(self, apply_softmax=False):
        """
        Init the class.

        Parameters:
        apply_softmax: Whether the loss function should first apply Softmax to the inputs.
        """
        self.apply_softmax = apply_softmax

    def __call__(self, y_pred: npt.ArrayLike, y_true: npt.ArrayLike) -> npt.ArrayLike:
        """
        Output the loss or cost calculated using y_pred and y_true.

        Parameters:
        y_pred: The predicted output of a layer.
        y_true: The True output it should have given.

        Return:
        The loss of this particular training example.
        """
        if self.apply_softmax:
            exps = np.exp(y_pred)
            exp_sums = np.sum(exps, axis=-1)
            exp_sums = np.repeat(
                exp_sums[:, np.newaxis], exps.shape[-1], axis=1)
            y_pred = exps / exp_sums
        losses = np.nan_to_num(-y_true * np.log(y_pred)
                               - (1 - y_true) * np.log(1 - y_pred))
        losses = np.sum(losses)
        return losses

    def _delta(self, z: npt.ArrayLike, y_pred: npt.ArrayLike,
               y_true: npt.ArrayLike, activation: Activation = None) -> npt.ArrayLike:
        """Find the derivative of the loss function with respect to y_true and y_pred."""
        return np.transpose(y_pred - y_true)
