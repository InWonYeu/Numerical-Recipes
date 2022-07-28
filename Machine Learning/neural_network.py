# Simple fully connected neural network Following the text book "Make Your Own Neural Network" by Tariq Rashid

import numpy as np


def tanh(x):
        return np.tanh(x)


def tanh_deriv(x):
        return 1.0 - np.tanh(x)**2


class NeuralNetwork:
    def __init__(self, structure, learning_rate):
        self.layers = len(structure)
        self.i_nodes = int(structure[0])
        self.h1_nodes = int(structure[1])
        self.h2_nodes = int(structure[2])
        self.o_nodes = int(structure[3])

        self.w_ih1 = np.random.normal(0.0, structure[0]**(-0.5), (self.h1_nodes, self.i_nodes))
        self.w_h1h2 = np.random.normal(0.0, structure[1]**(-0.5), (self.h2_nodes, self.h1_nodes))
        self.w_h2o = np.random.normal(0.0, structure[2]**(-0.5), (self.o_nodes, self.h2_nodes))

        self.act = tanh
        self.act_deriv = tanh_deriv
        self.lr = learning_rate

    def feedforward(self, input_list):
        inputs = (np.asfarray(input_list)).reshape([len(input_list),1])

        inputs_h1 = np.dot(self.w_ih1, inputs)
        outputs_h1 = self.act(inputs_h1)

        inputs_h2 = np.dot(self.w_h1h2, outputs_h1)
        outputs_h2 = self.act(inputs_h2)

        inputs_o = np.dot(self.w_h2o, outputs_h2)
        outputs_o = inputs_o

        return outputs_o

    def backprop(self, input_list, target_list):
        inputs = (np.asfarray(input_list)).reshape([len(input_list), 1])
        targets = (np.asfarray(target_list)).reshape([len(target_list), 1])

        inputs_h1 = np.dot(self.w_ih1, inputs)
        outputs_h1 = self.act(inputs_h1)

        inputs_h2 = np.dot(self.w_h1h2, outputs_h1)
        outputs_h2 = self.act(inputs_h2)

        inputs_o = np.dot(self.w_h2o, outputs_h2)
        outputs_o = inputs_o

        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        errors_o = outputs_o - targets
        errors_h2 = np.dot(self.w_h2o.transpose(), errors_o)
        errors_h1 = np.dot(self.w_h1h2.transpose(), errors_h2)

        # update the weights with the derivative (slope) of the loss function
        self.w_h2o -= self.lr * np.dot(errors_o, outputs_h2.transpose())
        self.w_h1h2 -= self.lr * np.dot(errors_h2 * self.act_deriv(inputs_h2), outputs_h1.transpose())
        self.w_ih1 -= self.lr * np.dot(errors_h1 * self.act_deriv(inputs_h1), inputs.transpose())


input_list = range(10)
target_list = [1.1]

NN = NeuralNetwork([10, 30, 30, 1], 0.1)
print("Input= ", input_list, "Output= ", "%4f" % NN.feedforward(input_list)[0, 0], " Target= ", target_list)

epochs = 30
for e in range(epochs):
    NN.backprop(input_list, target_list)
    print("Epoch= %4d" % (e + 1), "Output= ", "%4f" % NN.feedforward(input_list)[0, 0], " Target= ", target_list[0]) # test
