#!/usr/bin/env python

from layer import Layer

import jsonpickle
import json

class Network:
    
    def __init__(self, num_hidden_layers, num_test_feat, num_neurons_per_layer, num_possible_outputs):
        if num_hidden_layers > 1:
            self.hidden_layers = []
            self.hidden_layers.append(Layer(num_neurons_per_layer,num_test_feat))
            for i in range(num_hidden_layers-1):
                self.hidden_layers.append(Layer(num_neurons_per_layer,num_neurons_per_layer))
            self.hidden_layers.append(Layer(num_possible_outputs, num_neurons_per_layer))
        elif num_hidden_layers == 1:
            self.hidden_layers = []
            self.hidden_layers.append(Layer(num_possible_outputs, num_test_feat))
        
            
        
    def __str__(self):
        return json.dumps(json.loads(jsonpickle.encode(self.__dict__, unpicklable=False)), indent=4, sort_keys=True)
        
        
