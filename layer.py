#!/usr/bin/env python

from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, num_neurons_prev_layer):
        self.neurons = []
        for x in range(num_neurons):
            self.neurons.append(Neuron(num_neurons_prev_layer))
    
    def __str__(self):
        return json.dumps(self.__dict__, separators=(',', ': '), sort_keys=True)
