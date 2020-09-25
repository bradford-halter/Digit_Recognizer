#!/usr/bin/env python

def sig_func(x):
    return 1 / (1 + exp(-x))

def cross_prod(active_neuron, prev_layer):
    ret_val = 0
    
    for x in range(prev_layer):
        ret_val += active_neuron.weights[x] * prev_layer.neurons[x].alpha
        
    ret_val = sig_func(ret_val + bias)
    
    return ret_val
    
