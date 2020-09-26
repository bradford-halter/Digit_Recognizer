#I'm using this to test functions

from network import Network
from layer import Layer
from neuron import Neuron

from sig_func import *

import json.*

def test1():
    test_neuron = Neuron(2)
    test_layer = Layer(1, 2)
    test_layer.neurons[0] = 0
    test_layer.neurons[1] = 1
    test_neuron.weights[0] = 1
    test_neuron.weights[1] = 1
    
    print(sig_func(test_neuron, test_layer))

def main():
    test1()
