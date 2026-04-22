#!/usr/bin/env python3
"""Transfer learning on CIFAR-10 with MobileNetV2."""

import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """Preprocess CIFAR-10 data for MobileNetV2 transfer learning."""
    X_p = K.applications.mobilenet_v2.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)

    inputs = K.Input(shape=(32, 32, 3))
    resized = K.layers.Lambda(lambda x: tf.image.resize(x, (224, 224)))(inputs)

    base_model = K.applications.MobileNetV2(
        input_tensor=resized,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = K.layers.GlobalAveragePooling2D()(base_model.output)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        K.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    model.fit(
        X_train,
        Y_train,
        batch_size=64,
        epochs=20,
        validation_data=(X_valid, Y_valid),
        callbacks=callbacks,
        verbose=1
    )

    model.save('cifar10.h5')