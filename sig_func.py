#!/usr/bin/env python

def sigmoid_function(x):
    return 1 / (1 + exp(-x))

def sig_func(active_neuron, prev_layer):
    ret_val = 0
    
    for i in range(len(prev_layer)):
        ret_val += active_neuron.weights[i] * prev_layer.neurons[i].alpha
        
    ret_val = sig_func(ret_val + bias)
    
    return ret_val
    
