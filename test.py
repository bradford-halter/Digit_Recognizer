#I'm using this to test functions

from network import Network
from layer import Layer
from neuron import Neuron

from sig_func import *
from cost_func import *

import jsonpickle

def test1():
    test_neuron = Neuron(3)
    test_layer = Layer(3, 0)
    test_layer.neurons[0].alpha = 0
    test_layer.neurons[1].alpha = 1
    test_layer.neurons[2].alpha = 0
    test_neuron.weights[0] = 1
    test_neuron.weights[1] = 1
    test_neuron.weights[2] = 1
    
    print("This should print out 0.73:")
    print(sig_func(test_neuron, test_layer))
    
def test2():
    test_output_alphas = [0, 0]
    test_desired_alphas = [1, 0]
    
    print("This should print out 1:")
    print(cost_func(test_output_alphas, test_desired_alphas))
    
    test_output_alphas = [0, 0]
    test_desired_alphas = [1, 1]
    
    print("This should print out 2:")
    print(cost_func(test_output_alphas, test_desired_alphas))
    
    test_output_alphas = [0, 1]
    test_desired_alphas = [1, 0]
    
    print("This should print out 2:")
    print(cost_func(test_output_alphas, test_desired_alphas))
    
    test_output_alphas = [0.5, 1]
    test_desired_alphas = [1, 0]
    
    print("This should print out 1.25:")
    print(cost_func(test_output_alphas, test_desired_alphas))
    

def main():
    print("Starting test.main:")
    test2()

# Call the main() function when the program is started from command line.
if __name__ == "__main__":
    main()
