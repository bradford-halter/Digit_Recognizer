#!/usr/bin/env python

#Neuron class
#
#This contains a bias, an alpha value, and a set of weights based on the number of nodes in the previous layer.

class Neuron:
    def __init__(self, num_weights):
        self.alpha = 0
        self.bias = 0
        self.weights = []
        for i in range(num_weights):
            self.weights.append(0)
    
    def __str__(self):
        return json.dumps(self.__dict__, separators=(',', ': '), sort_keys=True)
