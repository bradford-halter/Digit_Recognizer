#!/usr/bin/env python

from sig_func import *

def propagation(network, input_layer):
    network.last_input_layer_used = input_layer # store a copy of the input layer, to be printed out layer
    for i in range(len(network.hidden_layers)):
        if i > 0:
            for j in range(len(network.hidden_layers[i].neurons)):
                cur_neuron = network.hidden_layers[i].neurons[j]
                cur_neuron.z, cur_neuron.alpha = sig_func(network.hidden_layers[i].neurons[j], network.hidden_layers[i - 1])
        else:
            for j in range(len(network.hidden_layers[i].neurons)):
                cur_neuron = network.hidden_layers[i].neurons[j]
                cur_neuron.z, cur_neuron.alpha = sig_func(network.hidden_layers[i].neurons[j], input_layer)
    
    print("propagation(): right before the end")
    alpha_values_for_output_layer = []
    for neuron in network.hidden_layers[-1].neurons:
        alpha_values_for_output_layer.append(neuron.alpha)
    
    return(alpha_values_for_output_layer)
    
def classify_propagation_output(network):
    confidence = -1
    output_index = -1
    
    for j in range(len(network.hidden_layers[-1].neurons)):
        if network.hidden_layers[-1].neurons[j].alpha > confidence:
            confidence = network.hidden_layers[-1].neurons[j].alpha
            output_index = j
            
    return [confidence, output_index]
