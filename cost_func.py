#!/usr/bin/env python

import math

def cost_func(output_layer_alphas, desired_alphas):
    ret_val = 0
    
    for i in range(len(output_layer_alphas)):
        ret_val += ((output_layer_alphas[i] - desired_alphas[i]) ** 2)
        
    return ret_val
