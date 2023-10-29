import numpy
import matplotlib.pyplot as plt


class NeuralNet:
    def __init__(self, inodes: int, hnodes: int, onodes: int, learning_rate: float):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        self.lr = learning_rate
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

    def print_weights(self):
        print(f"input to hidden weights {self.wih}")
        print(f"hidden to output weights {self.who}")
        # plt.imshow(self.wih, interpolation="nearest")
        # plt.show()  # Display the plot in a separate window

    def train(self):
        pass

    def query(self):
        pass
