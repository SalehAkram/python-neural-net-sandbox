from neuralnet import NeuralNet

neuralNet = NeuralNet(3, 3, 3, 0.3)

output = neuralNet.query([1.0, 0.5, -1.5])
print(output)
