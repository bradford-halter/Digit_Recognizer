#!/usr/bin/env python

from back_propagation import *
from cost_func import *

# 
# input_layer:  input for a single example
# desired_mat: output for a single example
#
def entire_network_back_prop(network, input_layer, cur_model_output, desired_mat): # cur_model_output is the current model outputs
    print("entire_network_back_prop: Begin")
    cost_ini = cost_func(cur_model_output, desired_mat)[2]
    print("entire_network_back_prop: finished calling cost_func()")
    new_cost = []
    
    if len(network.hidden_layers) == 1:
        print("entire_network_back_prop: len(network.hidden_layers) == 1")
        new_cost = back_propagation(network.hidden_layers[0], input_layer, cost_ini)
    else:
        for i in range(len(network.hidden_layers)):
            print("entire_network_back_prop: i == " + str(i))
            if i == 0:
                new_cost = back_propagation(network.hidden_layers[len(network.hidden_layers) - 1], network.hidden_layers[len(network.hidden_layers) - 2], cost_ini)
            elif i == len(network.hidden_layers) - 1:
                new_cost = back_propagation(network.hidden_layers[len(network.hidden_layers) - (i + 1)], input_layer, new_cost)
            else:
                new_cost = back_propagation(network.hidden_layers[len(network.hidden_layers) - (i + 1)], network.hidden_layers[len(network.hidden_layers) - (i + 2)], new_cost)

    print("entire_network_back_prop: Done")