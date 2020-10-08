#I'm using this to test functions

from network import Network
from layer import Layer
from neuron import Neuron

from sig_func import *
from cost_func import *
from propagation import *

import math
import csv
import random
import jsonpickle
#import matplotlib
import matplotlib.pyplot as plt

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
    
def test_exp_1():
    print("test_exp_1()")
    print(math.exp(1))
    
# For testing: Change the weights in each neuron to 1, and the bias to 0.
def initialize_network_for_testing(network, initial_weight = 1, initial_bias = 0):
    for cur_layer in network.hidden_layers:
        for cur_neuron in cur_layer.neurons:
            cur_neuron.bias = initial_bias
            for i in range(len(cur_neuron.weights)):
                cur_neuron.weights[i] = initial_weight
    
def test_cross_prod_1():
    '''
    # of hidden layer = 1
    # of test features = 4
    # of neurons per layer = 3
    # of possible outputs = 3
    
    Create an input layer that has 4 neurons.
    Have the alphas be in this order:
        0, 0.25, 0.5, 0.25
        
    Weights: Set to 1 for all neurons.
        
    Output: All of these neurons should be exactly the same, 
    and the output should be 0.73.
    
    Note: To run this test, the weights in each Neuron (in neuron.py) must default to a value of 1, and bias must default to 0.
    '''
    
    print("Started test_cross_prod_1()")
    
    alpha_values_for_input_layer = [0.0, 0.25, 0.5, 0.25]
    
    my_network = Network(1, 4, 3, 3)
    
    initialize_network_for_testing(my_network)
    
    # Create the input layer
    input_layer = Layer(4, 1)
    for i in range(len(input_layer.neurons)):
        input_layer.neurons[i].alpha = alpha_values_for_input_layer[i]
    
    print()
    print("Print the input_layer")
    print(input_layer)
    
    alpha_values_for_output_layer = propagation(my_network, input_layer)
    
    classification_values = classify_propagation_output(my_network)
    
    print()
    print("Output layer alpha values:")
    print(alpha_values_for_output_layer)
    
    print()
    print("Print out the classification values:")
    print(classification_values)
    
    print()
    print("Print out the network:")
    print(my_network)
    
    
    # All alpha values should be 0.73
    for alpha_val in alpha_values_for_output_layer:
        if abs(alpha_val - 0.73) > 0.01:
            print("Test failed :)")
            return
    
    print("Test succeeded! Amazing!")
    return

def test_forward_propagation_1_layers():
    print("test_forward_propagation_1_layers()")
    # Put in some examples where each example has the same output as input.
    # Train the neural network. The single weight should end up near 1 (?) and the bias near 0.
    # If the neural network ends up outputting a function like f(x) = x, then we're good.
    single_weight_and_bias = Network(1, 1, 1, 1)
    examples = [(x, x) for x in range(-10, 11)] # -10 to 10 (inclusive).
    
    # Set up an input layer for each example.
    examples_as_input_layers = [Layer(1, 1) for item in examples]
    
    inputs_of_network = [in_out_pair[0] for in_out_pair in examples]
    expected_outputs_of_network = normalize(inputs_of_network)
    
    #print("Print out the un-initiaized input layers that we will run evaluate the network on.")
    #for example_input_layer in examples_as_input_layers:
    #    print(example_input_layer)
    
    for i in range(len(examples)):
        # Set the "alpha" for the single neuron in each of the input layers 
        # to be equal to the example's input.
        # Sorry that is confusing. We are just setting up the inputs.
        examples_as_input_layers[i].neurons[0].alpha = examples[i][0] 
    
    outputs_of_network = []
    
    for input_layer in examples_as_input_layers:
        output_vector = propagation(single_weight_and_bias, input_layer)
        outputs_of_network.append(output_vector[0])
    
    print("These were the inputs: ")
    print(inputs_of_network)
    print()
    
    print("These were the outputs: ")
    print(["{:.2f}".format(output) for output in outputs_of_network])
    print()
    
    print("Let's print out the Network and see what happens.")
    print(single_weight_and_bias)
    print()
    
    print("Let's try to the output of our neural network before training, " \
           + "when evaluated on each input.")
    plt.plot(inputs_of_network, outputs_of_network, 'o', color='black');
    plt.show()
        
    print("OK some progress!")

def test_matplotlib_plot():
    x = [i for i in range(-10, 11)]
    y = [i for i in range(-10, 11)]
    plt.plot(x, y, 'o', color='black');
    plt.show()
    print("Booyah it printed.")

def main():
    # Set the random seed to 0 so tests will be repeatable.
    random.seed(0)
    
    print("Starting test.main:")
    #test1()
    #test2()
    #test_cross_prod_1()
    test_forward_propagation_1_layers()
    #test_matplotlib_plot()

# Call the main() function when the program is started from command line.
if __name__ == "__main__":
    main()
