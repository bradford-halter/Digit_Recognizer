#!/usr/bin/env python

from back_propagation import *
from cost_func import *

def entire_network_back_prop(network, input_layer, cost_mat, desired_mat): # cost_mat is the current model outputs
    cost_ini = cost_func(cost_mat, desired_mat)[2]
    new_cost = []
    
    if len(network.layers) == 1:
        new_cost = back_propagation(network.layers[0], input_layer, cost_ini)
    else:
        for i in range(len(network.layers)):
            if i == 0:
                new_cost = back_propagation(network.layers[len(network.layers) - 1], network.layers[len(network.layers) - 2], cost_ini)
            elif i == len(network.layers) - 1:
                new_cost = back_propagation(network.layers[len(network.layers) - (i + 1)], input_layer, new_cost)
            else:
                new_cost = back_propagation(network.layers[len(network.layers) - (i + 1)], network.layers[len(network.layers) - (i + 2)], new_cost)
