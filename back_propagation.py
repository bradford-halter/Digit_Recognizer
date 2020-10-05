#!/usr/bin/env python

from layer import Layer
from neuron import Neuron
from sig_func import *

def back_propagation(current_layer, previous_layer, cost_mat):
    #use this video for reference: https://www.youtube.com/watch?v=tIeHLnjs5U8&t=350s&ab_channel=3Blue1Brown
    #
    #create a matrix which represents the change of z_j^(L) w.r.t. weights_in_neurons^(L)
    
    
    new_weights = []
    new_bias = []
    new_cost = []

    #Calculates new weights and bias

    for i in range(len(current_layer.neurons)):
        new_del_z_w = 0
        
        for j in range(len(previous_layer.neurons)):
            new_del_z_w += previous_layer.neurons[j].alpha
            
        new_del_a = sig_func(current_layer.neurons[i], previous_layer)
        new_del_a *= (1 - sig_func(current_layer.neurons[i], previous_layer))
        
        new_del_C = 2 * cost_mat[i]
        
        new_weights.append(new_del_z_w * new_del_a * new_del_C)
        new_bias.append(new_del_a * new_del_C)
        
        new_del_z_a = 0
        
        for j in range(len(new_weights)):
            new_del_z_a += new_weights[j]

    # Update weights and bias in current layer

    for i in range(len(current_layer.neurons)):
        for j in range(len(current_layer.neurons[i].weights)):
            current_layer.neurons[i].weights[j] += new_weights[i]
        
        current_layer.neurons[i].bias += new_bias[i]
    
    # Create a new cost_mat

    for i in range(len(previous_layer.neurons)):
        cost_sum = 0
        for j in range(len(current_layer.neurons)):
            new_del_a = sig_func(current_layer.neurons[j], previous_layer)
            new_del_a *= (1 - sig_func(current_layer.neurons[j], previous_layer))
            
            new_del_C = 2 * cost_mat[j]
            
            cost_sum += new_del_a * new_del_C * current_layer.neurons[j].weights[i]
            
        new_cost.append(cost_sum)
    
    return new_cost
        
