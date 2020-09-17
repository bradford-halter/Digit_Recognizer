#!/usr/bin/env python

from layer import Layer

class Network:
    
    def __init__(self, num_layers, num_test_feat, num_neurons_per_layer, num_possible_outputs):
        self.output_layer = []
        for x in range(num_possible_outputs):
            self.output_layer.append(num_possible_outputs, num_neurons_per_layer)
        self.hidden_layers = []
        if num_layers > 0:
            self.hidden_layers.append(Layer(num_neurons_per_layer,num_test_feat))
            for x in range(num_layers-1):
                self.hidden_layers.append(Layer(num_neurons_per_layer,num_neurons_per_layer))
        
    def __str__(self):
        # Define what the print() function should do when passed a Network object.
        return json.dumps(self.__dict__, separators=(',', ': '), sort_keys=True)
        
