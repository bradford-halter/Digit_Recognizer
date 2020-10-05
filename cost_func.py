#!/usr/bin/env python

import math

def cost_func(output_layer_alphas, desired_alphas):
    
    ret_val_norm = 0
    ret_val_array = []
    ret_val_norm_array = []
    
    for i in range(len(output_layer_alphas)):
        ret_val_norm += ((output_layer_alphas[i] - desired_alphas[i]) ** 2)
        ret_val_array.append(output_layer_alphas[i] - desired_alphas[i]) 
        ret_val_norm_array.append((output_layer_alphas[i] - desired_alphas[i]) ** 2)
        
    return [ret_val_norm, ret_val_array, ret_val_norm_array]
