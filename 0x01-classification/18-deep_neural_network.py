#!/usr/bin/env python3
import numpy as np


class DeepNeuralNetwork:
   

    def __init__(self, nx, layers):
    
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        weights = {}
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            if i == 0:
                weights[key_w] = np.random.randn(layers[i], nx)*np.sqrt(2 / nx)
            else:
                weights[key_w] = np.random.randn(layers[i], layers[
                    i-1]) * np.sqrt(2 / layers[i-1])
            weights[key_b] = np.zeros((layers[i], 1))
        self.__weights = weights

    def forward_prop(self, X):
      
        # Input layer
        self.__cache['A0'] = X
        # Hidden and output layer
        for i in range(self.__L):
            # create keys to access weights(W), biases(b) and store in cache
            key_w = 'W' + str(i + 1)
            key_b = 'b' + str(i + 1)
            key_cache = 'A' + str(i + 1)
            key_cache_last = 'A' + str(i)
            # store activation in cache
            output_Z = np.matmul(self.__weights[key_w], self.__cache[
                key_cache_last]) + self.__weights[key_b]
            output_A = 1 / (1 + np.exp(-output_Z))
            self.__cache[key_cache] = output_A
        return output_A, self.__cache

    @property
    def cache(self):
     
        return self.__cache

    @property
    def L(self):
      
        return self.__L

    @property
    def weights(self):
      
        return self.__weights
