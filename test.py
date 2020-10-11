#I'm using this to test functions

from network import Network
from layer import Layer
from neuron import Neuron

from sig_func import *
from cost_func import *
from propagation import *
from back_propagation import *
from entire_network_back_prop import *

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
    plt.plot(inputs_of_network, outputs_of_network, 'o', color='red');
    plt.plot(inputs_of_network, expected_outputs_of_network, 'o', color='blue');
    plt.show()
        
    print("OK some progress!")

def test_cost_function_1_layers():
    print("test_cost_function_1_layers()")
    
    #--------------------------------------------
    # Just set up a network with 1 input 1 output
    #--------------------------------------------
    
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

    #--------------------------------------------
    # ^ Done setting up the network.
    #--------------------------------------------
    
    
    # Evaluate the networks on the inputs.
    outputs_of_network = []
    for input_layer in examples_as_input_layers:
        output_vector = propagation(single_weight_and_bias, input_layer)
        outputs_of_network.append(output_vector[0])
    
    # Calculate cost function.
    # We have a weird case where our inputs and outputs are both just scalars, so we need to put each one into a list first.
    output_layer_alphas = [ [x] for x in outputs_of_network ]
    desired_alphas =      [ [x] for x in expected_outputs_of_network ]
    
    the_costs = []
    for i in range(len(output_layer_alphas)):
        the_costs.append(cost_func(output_layer_alphas[i], desired_alphas[i]))
    # Now we have a list of the cost function output for each example.
    
    print("Cost values for 1st time we ran the network: ")
    for some_confusing_list in the_costs:
        print(some_confusing_list)
    print()
    
    # We need to takes the THIRD value from the_costs, and hand it off to backpropogation.
    ret_val_norm_array = [c for a, b, c in the_costs]
    print("ret_val_norm_array: ")
    print(ret_val_norm_array)
    print()
    
    costs_to_plot = [x[0] for x in ret_val_norm_array]
    
    # Now that we have the cost function values to hand off, lets sanity check them by plotting them.
    plt.plot(inputs_of_network, outputs_of_network,          'o', color='red');
    plt.plot(inputs_of_network, expected_outputs_of_network, 'o', color='blue');
    plt.plot(inputs_of_network, costs_to_plot,               'o', color='green');
    plt.show()
    
    print("The cost function looks accurate. Done for now.")
    
    
# Evaluate the network on the input, and get the outputs and costs back.
def test_back_propagation_1_layers_evaluate(examples_as_input_layers,
                                           single_weight_and_bias, 
                                           expected_outputs_of_network):

    # Evaluate the networks on the inputs.
    outputs_of_network = []
    for input_layer in examples_as_input_layers:
        output_vector = propagation(single_weight_and_bias, input_layer)
        outputs_of_network.append(output_vector[0])
    
    # Calculate cost function.
    # We have a weird case where our inputs and outputs are both just scalars, so we need to put each one into a list first.
    output_layer_alphas = [ [x] for x in outputs_of_network ]
    desired_alphas =      [ [x] for x in expected_outputs_of_network ]
    
    the_costs = []
    for i in range(len(output_layer_alphas)):
        the_costs.append(cost_func(output_layer_alphas[i], desired_alphas[i]))
    # Now we have a list of the cost function output for each example.
    
    #print("Cost values for 1st time we ran the network: ")
    #for some_confusing_list in the_costs:
    #    print(some_confusing_list)
    #print()
    
    # We need to takes the THIRD value from the_costs, and hand it off to backpropogation.
    # Each element of this list is a cost_mat to pass to backprop for 1 example.
    ret_val_norm_array = [c for a, b, c in the_costs]
    print("ret_val_norm_array: ")
    print(ret_val_norm_array)
    print()
    
    return outputs_of_network, ret_val_norm_array
    
    
def test_back_propagation_1_layers():
    print("test_back_propagation_1_layers()")
    
    #--------------------------------------------
    # Just set up a network with 1 input 1 output
    #--------------------------------------------
    
    # Put in some examples where each example has the same output as input.
    # Train the neural network. The single weight should end up near 1 (?) and the bias near 0.
    # If the neural network ends up outputting a function like f(x) = x, then we're good.
    single_weight_and_bias = Network(2, 1, 16, 1)
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

    #--------------------------------------------
    # ^ Done setting up the network.
    #--------------------------------------------
    
    costs_per_evaluation = []
    outputs_of_network_per_evaluation = []
    
    
    # Run back propagation once for each (input, expected_output) example.
    # We're not batching. Each time we call back_propagation(), 
    # it's doing backprop using 1 training example.
    for cur_num_evaluation in range(3):
    
        # Evaluate the network on the input, and get the outputs and costs.
        outputs_of_network, ret_val_norm_array =  \
        test_back_propagation_1_layers_evaluate(examples_as_input_layers, 
                                            single_weight_and_bias, 
                                            expected_outputs_of_network)
        # Store the outputs and costs that we just calculated by running the network.
        outputs_of_network_per_evaluation.append(outputs_of_network)
        costs_per_evaluation.append(ret_val_norm_array)
    
        # Do backprop across all (input, output) examples 1 time.
        for i in range(len(examples_as_input_layers)):
            cur_desired_output = [expected_outputs_of_network[i]] # The expected output for this 1 input example. Included within an array if it is a scalar value.
            cur_model_output = [outputs_of_network_per_evaluation[cur_num_evaluation][i]]
            entire_network_back_prop(single_weight_and_bias, examples_as_input_layers[i], cur_model_output, cur_desired_output)
            #new_costs = back_propagation(single_weight_and_bias.hidden_layers[0], 
            #                             examples_as_input_layers[i], 
            #                             costs_per_evaluation[cur_num_evaluation][i])
        # TODO: there is a bug w/ list index out of range. Handle it using the new_costs variable, somehow.
    
    #--------------------------------------------
    # ^ Done with backprop. Now print results.
    #--------------------------------------------
    
    # Evaluate the network on the inputs a final time, and see difference in output and costs.
    outputs_of_network_final_time, ret_val_norm_array_final_time =  \
    test_back_propagation_1_layers_evaluate(examples_as_input_layers, 
                                        single_weight_and_bias, 
                                        expected_outputs_of_network)

    costs_to_plot             = [x[0] for x in costs_per_evaluation[0]]
    costs_to_plot_final_time  = [x[0] for x in ret_val_norm_array_final_time]
    
    
    print("Evalute the network on the inputs a final time, and see difference in output after backprop.")
    print("1st overall cost: ", end="")
    print(sum(costs_to_plot))
    print("2nd overall cost: ", end="")
    print(sum(costs_to_plot_final_time))
    
    original_output_of_network = outputs_of_network_per_evaluation[0]
    
    plt.plot(inputs_of_network, original_output_of_network,     'o', color='red');
    plt.plot(inputs_of_network, expected_outputs_of_network,    'o', color='blue');
    plt.plot(inputs_of_network, outputs_of_network_final_time,  'o', color='purple');
    plt.plot(inputs_of_network, costs_to_plot,              'o', color='green');
    plt.plot(inputs_of_network, costs_to_plot_final_time,   'o', color='yellow');
    plt.show()
    

    
    print("Done.")

# Simplest MNIST Test 1:
#
#784 inputs.
#10 outputs.
#doesnt matter how many neurons per layer
#1 hidden layer.
#
#    In this test there will be 10 outputs, each with 784 weights, + 1 for the bias.
#    
#Train with ONLY 1 example, and expect classification to work for only that 1 example.
#    e.g. do backprop with an image of a 1, and the output classification 1. And then train the network to get that right.
#
# Pseudocode:
#Open the CSV and read in 1 example.
#    The 2nd row of the CSV file.
#    
#Set up 1 input layer with 784 values in the layer.
#
#Create MNIST outputs
#    Set up an array of 10 outputs containing floats in [0.0, 1.0].
#        The value at index 1 should be 1.0, all else 0.0.
#        
#        
#In a loop: (For some num iterations)
#    Print the iteration num.
#    # Network.train() is for 1 iteration
#    cur_forward_prop_result, cur_cost_function_result = Network.train(example_inputs, example_outputs) 
#    train():
#        Do forward propagation.
#        Store the output of forward prop.
#            
#        Call entire_network_back_prop().
#            Print out the forward prop results.
#            Print out current value of cost function.
#
def simple_mnist_test_1():
    
    single_training_example = None

    # Open the CSV file as read-only. Python will close it when the program exits.
    with open('train_small.csv', 'r') as csvfile: 

        training_data = csv.reader(csvfile)

        # The first row in the training data CSV file is text, the names of each column.
        csv_column_names = next(training_data)
        
        single_training_example = next(training_data)
        
    example_inputs  = []
    example_outputs = []
    
    cur_inputs_str = single_training_example[1:]
    cur_inputs_int = normalize([ int(x) for x in cur_inputs_str ])
    
    example_inputs.append(cur_inputs_int) # 
    example_outputs.append( [0.0 for x in range(10)] )
    example_outputs[0][ int(single_training_example[0]) ] = 1.0
    
    my_network = Network(2, 784, 1, 10)
    
    cur_model_output         = None
    cur_cost_function_result = None
    
    num_iterations_of_backprop = 10
    for i in range(num_iterations_of_backprop):
        cur_cost_function_result = my_network.train(example_inputs, example_outputs)
        print("cur_cost_function_result: " + str(cur_cost_function_result))
        print()
    
    print("Holy guac, did it work??")
    print()
    print("simple_mnist_test_1: Done")

# MNIST Test 2:
#
# 
#
def simple_mnist_test_2():
    
    training_examples = []

    # Open the CSV file as read-only. Python will close it when the program exits.
    with open('train_small.csv', 'r') as csvfile: 

        training_data = csv.reader(csvfile)

        # The first row in the training data CSV file is text, the names of each column.
        csv_column_names = next(training_data)
        
        for row in training_data:
            training_examples.append(row)
        
    example_inputs  = []
    example_outputs = []
    
    for cur_training_example in training_examples:
        cur_inputs_str = cur_training_example[1:]
        cur_inputs_int = normalize([ int(x) for x in cur_inputs_str ])
        
        example_inputs.append(cur_inputs_int) # 
        example_outputs.append( [0.0 for x in range(10)] )
        example_outputs[0][ int(cur_training_example[0]) ] = 1.0
    
    my_network = Network(2, 784, 4, 10)
    
    cur_model_output         = None
    cur_cost_function_result = None
    
    num_iterations_of_backprop = 10
    for i in range(num_iterations_of_backprop):
        cur_cost_function_result = my_network.train(example_inputs, example_outputs)
        print("cur_cost_function_result: " + str(["{:.4f}".format(x) for x in cur_cost_function_result]))
        print()


def test_normalize():
    print("test_normalize()")
    x = [i * 1000 for i in range(-5, 6)]
    print("in:  ", end="")
    print(x)
    y = normalize(x)
    print("out: ", end="")
    print(y)

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
    #test_forward_propagation_1_layers()
    #test_cost_function_1_layers()
    #test_back_propagation_1_layers()
    #test_matplotlib_plot()
    #test_normalize()
    #simple_mnist_test_1()
    simple_mnist_test_2()

# Call the main() function when the program is started from command line.
if __name__ == "__main__":
    main()
