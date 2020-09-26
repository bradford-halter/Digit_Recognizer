#!/usr/bin/env python

#Neuron class
#
#This contains a bias, an alpha value, and a set of weights based on the number of nodes in the previous layer.

import jsonpickle

class Neuron:
    def __init__(self, num_weights):
        self.alpha = random.random()
        self.bias = random.random()
        self.weights = []
        for i in range(num_weights):
            self.weights.append(random.random())
    
    def __str__(self):
        return jsonpickle.encode(self.__dict__, unpicklable=False)
