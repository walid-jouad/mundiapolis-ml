#!/usr/bin/env python3
""" NeuronClass """

import numpy as np


class Neuron:
    """ ClassNeuron initializer """

    def __init__(self, nx):
        """ Neuron fonction """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.b = 0
        self.A = 0
        self.W = np.random.randn(nx).reshape(1, nx)
