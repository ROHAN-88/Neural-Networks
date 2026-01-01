import numpy as np
from cd import create_data

X,y = create_data(8,3)

class Layer_Dense:
    def __init__(self,n_inputs,n_nureons):
        self.weigths =0.10* np.random.rand(n_inputs,n_nureons)
        self.biaes = np.zeros((1,n_nureons))
        
    def forward(self,inputs): #inputs is data you provide
        self.output = np.dot(inputs,self.weigths) + self.biaes

class Activaion_ReLU:
    def forward(self,input):
        self.output = np.maximum(0,input)

class Softmax_Activation:
    def forward(self,inputs):
        exp_value =  np.exp(inputs - np.max(input,axis=1,keepdims=True)) # this exponential Funtion np.exp() is numpy function for expoential Funtions 
        probabilities = exp_value / np.sum(exp_value,axis=1 , keepdims=True) # This is normalized value process
        self.output = probabilities



layer1 = Layer_Dense(2,8)
activaion1 = Activaion_ReLU()

activaion2= Softmax_Activation()

layer1.forward(X)
