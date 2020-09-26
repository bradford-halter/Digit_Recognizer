#I'm using this to test functions

from network import Network
from layer import Layer
from neuron import Neuron

from sig_func import *

import jsonpickle

def test1():
    test_neuron = Neuron(2)
    test_layer = Layer(2, 0)
    test_layer.neurons[0].alpha = 0
    test_layer.neurons[1].alpha = 1
    test_neuron.weights[0] = 1
    test_neuron.weights[1] = 1
    
    print("This should print out 0.73:")
    print(sig_func(test_neuron, test_layer))

def main():
    print("Starting test.main:")
    test1()

# Call the main() function when the program is started from command line.
if __name__ == "__main__":
    main()
