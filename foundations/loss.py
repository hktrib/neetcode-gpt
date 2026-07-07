import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # y_pred += 1e-7
        loss= -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.round(loss, 4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1- epsilon)

        # axis == 1 ensures that we calculate the loss of the prediction for each sample
        # and then compare indivial sample loss values summed and the mean of all those summed values
        # across all samples to the one-hot encoded true labels. 

        # if we don't keep axis == 1 then we just sum all loss values across all true labels
        # the mean of a scalar will remain as it is and loss will scale up proportionately with the no of 
        # samples
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        return np.round(loss, 4)
