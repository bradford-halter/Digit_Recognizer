#!/usr/bin/env python

def propagation(network, input_layer):
    network.hidden_layers.insert(0, input_layer)
    
    for i in range(1,len(network.hidden_layers)):
        for j in range(len(network.hidden_layers[i].neurons)):
            network.hidden_layers[i].neurons[j].alpha = sig_func(network.hidden_layer[i].neurons[j], network.hidden_layers[i - 1])
    
    confidence = -1
    output_index = -1
    
    for j in range(len(network.hidden_layers[-1].neurons)):
        if network.hidden_layers[-1].neurons[j].alpha > confidence:
            confidence = network.hidden_layers[-1].neurons[j].alpha
            output_index = j
            
    return [confidence, output_index]
