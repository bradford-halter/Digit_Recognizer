#!/usr/bin/env python

from neuron import Neuron

import jsonpickle

class Layer:
    def __init__(self, num_neurons, num_neurons_prev_layer):
        
        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(num_neurons_prev_layer))
        # These are used just for caching propogation/backpropogation results:
        self.weights = []
        self.alphas = []
        self.biases = []
    # y: For the very last layer (the output layer) in the network,
    #    y is the provided labelled data.
    def backpropogate(self, previous_layer, y):
        refresh_alpha_list()
        refresh_bias_list()
        self.backpropogate_weights(previous_layer, y)
        self.backpropogate_bias(previous_layer, y)
        print("backpropogate: It's still TODO boi!")
    
    def backpropogate_weights(self, previous_layer, y):
        previous_layer.refresh_alpha_list()
        delta_c_over_delta_w_l = []
        deriv_sig_z = [derivative_of_sigmoid(x) for x in self.z]
        for i in range(len(neurons)):
            output = previous_layer.alphas[i] * deriv_sig_z[i] * 2 * (self.alphas[i] - y[i])
            delta_c_over_delta_w_l.append(output)
        return(delta_c_over_delta_w_l)
        
    
    def __str__(self):
        return jsonpickle.encode(self.__dict__, unpicklable=False)
        
    def __len__(self):
        return len(self.neurons)

    def refresh_alpha_list(self):
        self.alphas = [neuron.alpha for neuron in self.neurons]
        
    def refresh_weights_list(self):
        print("don't even need this lol")
    
    def refresh_bias_list(self):
        self.biases = [neuron.bias for neuron in self.neurons]
    
    def multiply(self, lista, listb):
        [a*b for a,b in zip(lista, listb)]
