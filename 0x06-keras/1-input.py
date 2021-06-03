#!/usr/bin/env python3

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
 
    K.Model()
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=K.regularizers.l2(lambtha))(inputs)
    y = x
    rate = 1 - keep_prob
    for i in range(1, len(layers)):
        if i == 1:
            y = K.layers.Dropout(rate)(x)
        else:
            y = K.layers.Dropout(rate)(y)
        y = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lambtha))(y)
    model = K.Model(inputs=inputs, outputs=y)
    return model
