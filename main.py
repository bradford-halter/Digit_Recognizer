#!/usr/bin/env python

# Python 3.6

from network import Network
from neuron import Neuron
from layer import Layer

from propagation import *

import math
import csv
import random

def test_exp_1():
    print("test_exp_1()")
    print(math.exp(1))
 
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
    
    Note: To run this test, the weights in each Neuron (in neuron.py) must default to a value of 1.
    '''
    
    print("Started test_cross_prod_1()")
    
    alpha_values_for_input_layer = [0.0, 0.25, 0.5, 0.25]
    
    my_network = Network(3, 4, 3, 3)
    
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
            return "Test failed :)"
    
    return "Test succeeded! Amazing!"
    
 
def main():
    # Set the random seed to 0, so that tests of the code will be repeatable.
    random.seed(0)
    
    # Open the CSV file as read-only. Python will close it when the program exits.
    with open('train_small.csv', 'r') as csvfile: 

        training_data = csv.reader(csvfile)

        # The first row in the training data CSV file is text, the names of each column.
        csv_column_names = next(training_data)
        
        # Print the column names from the 1st line of the CSV file.
        #print(csv_column_names)

        # Iterate over each row of the training data.
        for row in training_data:
            #print(row)
            # Let's be real, we don't want to print this much stuff.
            break
            
        print(test_cross_prod_1())
        test_exp_1()
            
        # Exit the program.
        print("Done.")
        return
            

# Call the main() function when the program is started from command line.
if __name__ == "__main__":
    main()