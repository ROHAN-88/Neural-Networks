import math
import numpy as np

layer_outputs =[ [4.8,1.21,2.385],
                [8.9,-1.81,0.2],
                [1.41,1.051,0.026]]

#Exponential funtions helps so  that output is not negative without losing the meaning of negative
#E= 2.71828182846
# E = math.e

# exp_value = []

# for output in layer_outputs:
#     exp_value.append(E**output)


#----------- the above exp value done easer using numpy
exp_value = np.exp(layer_outputs)


# Now the normalization is done .It is done after exponentioal funtions

# norm_base = sum(exp_value)
# norm_value = []

# for values in exp_value:
#     norm_value.append(values / norm_base)

#----------- the above normalization value done easer using numpy
# norm_values = exp_value / np.sum(exp_value)b 

# print("this is norm value" , norm_values)

norm_value = exp_value / np.sum(exp_value,axis=0,keepdims=True)

print(norm_value)