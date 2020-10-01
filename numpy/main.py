#!/usr/bin/env python
# Python 3.6

import random
import numpy

def main():
    # Set the random seed to 0, so that tests of the code will be repeatable.
    random.seed(0)
    
    a_random_vector = numpy.random.uniform(size=10)
    print(a_random_vector)

# Call the main() function when the program is started from command line.
if __name__ == "__main__":
    main()