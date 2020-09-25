#!/usr/bin/env python

from layer import Layer

# Modify network.py

class Network:
    
    def __init__(self, num_hidden_layers, num_test_feat, num_neurons_per_layer, num_possible_outputs):
        if num_hidden_layers > 1:
            self.output_layer = []
            self.output_layer.append(Layer(num_possible_outputs, num_neurons_per_layer))
            self.hidden_layers = []
            self.hidden_layers.append(Layer(num_neurons_per_layer,num_test_feat))
            for i in range(num_hidden_layers):
                self.hidden_layers.append(Layer(num_neurons_per_layer,num_neurons_per_layer))
        elif num_hidden_layers == 1:
            self.output_layer = []
            self.output_layer.append(Layer(num_possible_outputs, num_test_feat))
        
            
        
    def __str__(self):
        # Define what the print() function should do when passed a Network object.
        return json.dumps(self.__dict__, separators=(',', ': '), sort_keys=True)
        
