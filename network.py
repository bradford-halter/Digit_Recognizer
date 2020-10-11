#!/usr/bin/env python

from layer import Layer

import jsonpickle
import json

class Network:
    
    # Terminology: 1 hidden layer means there's only an output layer.
    def __init__(self, num_hidden_layers, num_test_feat, num_neurons_per_layer, num_possible_outputs):
        if num_hidden_layers > 1:
            self.hidden_layers = []
            self.hidden_layers.append(Layer(num_neurons_per_layer,num_test_feat))
            for i in range(num_hidden_layers-2):
                self.hidden_layers.append(Layer(num_neurons_per_layer,num_neurons_per_layer))
            self.hidden_layers.append(Layer(num_possible_outputs, num_neurons_per_layer))
        elif num_hidden_layers == 1:
            self.hidden_layers = []
            self.hidden_layers.append(Layer(num_possible_outputs, num_test_feat))
        
        # We do not use this for calculation. When we call the propagation function,
        # we store a copy of the input layer here, so that it may be printed out later.
        self.last_input_layer_used = [] 
            
        
    def __str__(self):
        return json.dumps(json.loads(jsonpickle.encode(self.__dict__, unpicklable=False)), indent=4, sort_keys=True)
        
#
#   example_inputs:  inputs from the training data. 
#       This is a list of lists, where each inner list contains 1 value for each feature in 1st layer of the network.
#       The value of each input is in range [0.0, 1.0].
#   example_outputs: outputs from the training data. The dimension of this list must match that of example_inputs.
#       The value of each output is in range [0.0, 1.0].
#       
#   Network.train() is for 1 iteration of forward prop, calculating cost, and then back prop.
#
# pseudocode:
#   for each pair of input/output example:
#       As needed, create Layer classes to hold the input_layer for each example input.
#       Do forward propagation.
#       Store the output of forward prop.         
#       Call entire_network_back_prop().
    def train(self, example_inputs, example_outputs):
        for cur_input, cur_output in zip(example_inputs, example_outputs):
            print(cur_input, cur_output)
            
    
    
    
    
    
    