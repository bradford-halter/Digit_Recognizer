#!/usr/bin/env python
#
# Call sig_func for the correct value

import math

def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))

def sig_func(active_neuron, prev_layer):
    ret_val1 = 0
    
    for i in range(len(prev_layer.neurons)):
        ret_val1 += active_neuron.weights[i] * prev_layer.neurons[i].alpha
        
    ret_val2 = sigmoid_function(ret_val1 + active_neuron.bias)
    
    return [ret_val1, ret_val2]
    
