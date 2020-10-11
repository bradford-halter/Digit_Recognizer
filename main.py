#!/usr/bin/env python

# Python 3.6

from network import Network
from neuron import Neuron
from layer import Layer

import jsonpickle
 
def main():
    # Set the random seed to 0, so that tests of the code will be repeatable.
    random.seed(0)
    
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
    
    cur_cost_function_result = None
    
    cur_example_num = 0
    step_size = 20 # number of examples to train on in between saves.
    while cur_example_num + step_size < len(training_examples) - 1
        cur_cost_function_result = my_network.train(
                                                 example_inputs[cur_example_num : cur_example_num + step_size], 
                                                example_outputs[cur_example_num : cur_example_num + step_size])
        print("cur_cost_function_result: " + str(["{:.4f}".format(x) for x in cur_cost_function_result]))
        print()
        
        # Save the model to JSON format where it can be restored later.
        model_as_json = jsonpickle.encode(my_network)
        with open('model.json', 'w') as jsonpickle_savefile:
            jsonpickle_savefile.write(model_as_json)
        
        cur_example_num += 20 
            
    # Exit the program.
    print("Done.")
    return
            

# Call the main() function when the program is started from command line.
if __name__ == "__main__":
    main()