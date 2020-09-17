#Neuron class
#
#This contains a bias, an alpha value, and a set of weights based on the number of nodes in the previous layer.

class neuron:
   def __init__(self, num_weights):
      self.alpha = 0
      self.bias = 0
      self.weights = []
      for x in range(num_weights):
         self.weights.append(0)
