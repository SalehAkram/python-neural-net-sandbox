import numpy
from scipy.special import expit as sigmoid
from neuralnets.neuralnet import NeuralNet


class NeuralNetMultiLayers(NeuralNet):

    def __init__(self, inodes, hnodes_list, onodes, learning_rate):
        self.inodes = inodes
        self.hnodes_list = hnodes_list
        self.onodes = onodes
        self.lr = learning_rate
        self.hyper_parameters_dic = {
            "on": self.onodes,
            "in": self.inodes
        }
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = []

        # Initialize weights for input to first hidden layer
        weights.append(numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes_list[0], self.inodes)))

        # Initialize weights for hidden layer to hidden layer
        for i in range(1, len(self.hnodes_list)):
            weights.append(numpy.random.normal(0.0, pow(self.hnodes_list[i - 1], -0.5),
                                               (self.hnodes_list[i], self.hnodes_list[i - 1])))

            # Initialize weights for the last hidden layer to output
            weights.append(numpy.random.normal(0.0, pow(self.hnodes_list[-1], -0.5),
                                               (self.onodes, self.hnodes_list[-1])))

        return weights

    def activation_function(self, x):
        return sigmoid(x)

    def hyper_parameters(self, key):
        return self.hyper_parameters_dic[key]

    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        layer_outputs = [inputs]
        errors = []

        # Forward pass
        for i in range(len(self.weights)):
            layer_inputs = numpy.dot(self.weights[i], layer_outputs[-1])
            layer_outputs.append(self.activation_function(layer_inputs))

        # Calculate the error for the output layer
        output_errors = targets - layer_outputs[-1]
        errors.append(output_errors)

        # Backpropagation
        for i in range(len(self.weights) - 1, 0, -1):
            hidden_errors = numpy.dot(self.weights[i].T, errors[0])
            errors.insert(0, hidden_errors)

        for i in range(len(self.weights)):
            delta = errors[i] * layer_outputs[i + 1] * (1 - layer_outputs[i + 1])
            weight_update = self.lr * numpy.dot(delta, layer_outputs[i].T)
            self.weights[i] += weight_update

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        layer_outputs = [self.activation_function(numpy.dot(self.weights[0], inputs))]

        for i in range(1, len(self.weights)):
            layer_outputs.append(self.activation_function(numpy.dot(self.weights[i], layer_outputs[i - 1])))

        return layer_outputs[-1]
