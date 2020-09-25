#!/usr/bin/env python

from neuron import Neuron

import jsonpickle

class Layer:
    def __init__(self, num_neurons, num_neurons_prev_layer):
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(num_neurons_prev_layer))
    
    def __str__(self):
        return jsonpickle.encode(self.__dict__, unpicklable=False)
