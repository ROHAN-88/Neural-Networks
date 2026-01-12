import numpy as np 
from cd import create_data


class Layer__Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs , n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        
class Activation_ReLu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_softmax:
    def forward(self,inputs):
        exp_value = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))   
        probabilities =exp_value/ np.sum(exp_value,axis=1,keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7 , 1-1e-7)

        if len(y_true.shape) == 1: # for one deminsion Array formate
            correct_confidences = y_pred_clipped[range(samples),y_true ]
        elif len(y_true.shape) == 2: # for 2 dimension Array Formats
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

 
X,y = create_data(points= 100,classes=3)

dense1 = Layer__Dense(2,3)
activaion1 = Activation_ReLu()

dense2 = Layer__Dense(3,7)
activaion2 = Activation_softmax()

dense1.forward(X)
activaion1.forward(dense1.output)

dense2.forward(activaion1.output)
activaion2.forward(dense2.output)

# print(activaion2.output[:5])


loss_funtion = Loss_CategoricalCrossentropy()
loss = loss_funtion.calculate(activaion2.output,y)

print("loss",loss)