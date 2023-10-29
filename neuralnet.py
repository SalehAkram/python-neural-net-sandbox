import numpy
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid


class NeuralNet:
    def __init__(self, inodes: int, hnodes: int, onodes: int, learning_rate: float):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        self.lr = learning_rate
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: sigmoid(x)

    def print_weights(self):
        print(f"input to hidden weights {self.wih}")
        print(f"hidden to output weights {self.who}")
        # plt.imshow(self.wih, interpolation="nearest")
        # plt.show()  # Display the plot in a separate window

    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
