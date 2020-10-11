#!/usr/bin/env python

# Python 3.6

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
import csv

def main():
    print("Start")
    
    #--------------------------------------------
    # Read in the test inputs
    #--------------------------------------------
    
    # Set the random seed to 0, so that tests of the code will be repeatable.
    random.seed(0)
    
    testing_examples = []

    # Open the CSV file as read-only. Python will close it when the program exits.
    with open('test.csv', 'r') as csvfile: 

        testing_data = csv.reader(csvfile)

        # The first row in the testing data CSV file is text, the names of each column.
        csv_column_names = next(testing_data)
        
        for row in testing_data:
            testing_examples.append(row)
        
    test_inputs  = []
    
    for cur_testing_example in testing_examples:
        cur_inputs_str = cur_testing_example
        cur_inputs_int = normalize([ float(x) for x in cur_inputs_str ])
        test_inputs.append(cur_inputs_int) 
    
    #--------------------------------------------
    # Load the model
    #--------------------------------------------
    
    my_network = None
    with open('model.json', 'r') as unpicklejar:
        my_network = jsonpickle.decode(unpicklejar.read())
    if my_network == None:
        raise Exception("model.json was not loaded.")
    
    #--------------------------------------------
    # Get each classification from model and save.
    #--------------------------------------------
    
    # Open the CSV file to write classifications
    with open('test_set_classifications.csv', mode='w', newline='') as test_set_classifications_file:
        out_writer = csv.writer(test_set_classifications_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for i, cur_input in enumerate(test_inputs):
            cur_input_classification = my_network.classify(cur_input)
            out_writer.writerow([i+1, cur_input_classification])
            
            print(f'{i+1}, {cur_input_classification}')
        

    print("Done.")
    
# Call the main() function when the program is started from command line.
if __name__ == "__main__":
    main()