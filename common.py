from typing import Optional

import numpy as np
import keras
import keras.backend as K


def set_model_weights_to_unity(model: keras.Model) -> None:
    """Set all weights of an arbitrary Keras model to one."""
    shapes = (layer.shape for layer in model.get_weights())
    model.set_weights([np.ones(shape) for shape in shapes])


def get_gradient(model: keras.Model):
    """Taken from https://github.com/keras-team/keras/issues/2226#issuecomment-381807035"""
    weights = model.trainable_weights
    gradients = model.optimizer.get_gradients(model.total_loss, weights)
    input_tensors = (
        model.inputs + model.sample_weights + model.targets + [K.learning_phase()]
    )
    return K.function(inputs=input_tensors, outputs=gradients)


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    mask: bool,
) -> float:
    """Compute the mean absolute error for a number of predictions."""
    mask_indices = np.where(y_pred != 0.0) if mask else ...
    return np.average(np.abs(y_true - y_pred)[mask_indices], weights=weights)
