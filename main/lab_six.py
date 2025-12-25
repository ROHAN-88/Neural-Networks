import math

layer_outputs = [4.8,1.21,2.385]

#E= 2.71828182846
E = math.e

exp_value = []

for output in layer_outputs:
    exp_value.append(E**output)

print(exp_value)