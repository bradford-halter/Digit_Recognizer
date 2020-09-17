#!/usr/bin/env python

from . import neuron

class layer:
   def __init__(self, num_neurons, num_neurons_prev_layer):
      self.neurons = []
      for x in range(num_neurons):
         self.neurons.append(neuron(num_neurons_prev_layer))
