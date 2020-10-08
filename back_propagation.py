#!/usr/bin/env python

from sig_func import *

def back_propagation(current_layer, previous_layer, cost_mat):
    #use this video for reference: https://www.youtube.com/watch?v=tIeHLnjs5U8&t=350s&ab_channel=3Blue1Brown
    #
    
    
    new_weights = []
    new_bias = []
    new_cost = []

    #Calculate new weights and bias
    for i in range(len(current_layer.neurons)):
        new_del_z_w = 0
        
        #these derivatives are found in the video above at 8:27
        
        #this is the derivative of the z function at the top right
        for j in range(len(previous_layer.neurons)):
            new_del_z_w += previous_layer.neurons[j].alpha
        
        #derivative of the sigmoid function after the z function
        new_del_a = sig_func(current_layer.neurons[i], previous_layer)
        new_del_a *= (1 - sig_func(current_layer.neurons[i], previous_layer))
        
        #this is the derivative of the cost function after the sigmoid function
        new_del_C = 2 * cost_mat[i]
        
        #storing the new weights and bias' to update later
        new_weights.append(new_del_z_w * new_del_a * new_del_C)
        new_bias.append(new_del_a * new_del_C)
        
    # Update weights and bias in current layer
    for i in range(len(current_layer.neurons)):
        for j in range(len(current_layer.neurons[i].weights)):
            current_layer.neurons[i].weights[j] -= new_weights[i]
        
        current_layer.neurons[i].bias -= new_bias[i]
    
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
        
