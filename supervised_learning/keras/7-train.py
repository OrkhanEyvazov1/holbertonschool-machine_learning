#!/usr/bin/env python3
"""7-train.py
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=0.1, verbose=True, shuffle=False):
    """train_model - trains a model using mini-batch gradient descent
    Args:
        network: is the model to train
        data: is a numpy.ndarray of shape (m, nx) containing the input data
            where m is the number of data points and nx is the number of
            features
        labels: is a one-hot numpy.ndarray of shape (m, classes) containing
            the labels of data where classes is the number of classes
        batch_size: is the number of data points in a batch
        epochs: is the number of passes through data for mini-batch gradient
            descent
        validation_data: is the data to validate the model with, if not None
            the model should be validated against this data at the end of
            each epoch. validation_data is a tuple (val_data, val_labels)
            where val_data is the input data to validate the model with and
            val_labels are the labels of val_data
        early_stopping: is a boolean that indicates whether early stopping
            should be used. Early stopping is a technique used to stop training
            once the validation loss does not improve for a certain number of
            consecutive epochs, called patience
        patience: is the patience used for early stopping
        learning_rate_decay: is a boolean that indicates whether learning rate decay
            should be used
        alpha: is the initial learning rate
        decay_rate: is the decay rate
    Returns:
        the History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience))
    if learning_rate_decay and validation_data:
        K.backend.set_value(network.optimizer.learning_rate, alpha)

        def decay_lr(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_callback = K.callbacks.LearningRateScheduler(decay_lr, verbose=1)
        callbacks.append(lr_callback)
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
