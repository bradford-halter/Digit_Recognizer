#!/usr/bin/env python

# Python 3.6

#from . import Network
#from . import Neuron
#from . import Layer

import csv

def main():
    # Open the CSV file as read-only. Python will close it when the program exits.
    with open('train_small.csv', 'r') as csvfile: 

        training_data = csv.reader(csvfile)

        # The first row in the training data CSV file is text, the names of each column.
        csv_column_names = next(training_data)
        
        # Print the column names from the 1st line of the CSV file.
        print(csv_column_names)

        # Iterate over each row of the training data.
        for row in training_data:
            print(row)
            # Let's be real, we don't want to print this much stuff.
            break
            
        # Exit the program.
        print("Done.")
        return
            

# Call the main() function when the program is started from command line.
if __name__ == "__main__":
    main()