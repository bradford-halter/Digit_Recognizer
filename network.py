#!/usr/bin/env python

from . import Neuron
from . import Layer

class Network:
    
    def __init__(self, num_hidden_layers, num_neurons_per_layer):
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        
    def __str__(self):
        # Define what the print() function should do when passed a Network object.
        return "TODO"
        