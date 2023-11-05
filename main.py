from neuralnet import NeuralNet
from neuralnetv2 import NeuralNetV2
import numpy
import matplotlib.pyplot as plt

output_nodes = 10
input_nodes = 784
hidden_nodes = 100
learning_rate = 0.1
neural_net = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

# neural net v2 allows for multiple hidden layers, example [100,50] = two hidden layers first layer has 100 nodes
# 2nd layer has 50 nodes
hidden_layers = [100, 50]
neural_net_v2 = NeuralNetV2(input_nodes, hidden_layers, output_nodes, learning_rate)
# train the neural net
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 2
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(",")
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        neural_net.train(inputs, targets)
        # neural_net_v2.train(inputs,targets)

# test the neural net
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
score_card = []

for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = neural_net.query(inputs)
    # outputs = neural_net_v2.query(inputs)
    label = numpy.argmax(outputs)
    if label == correct_label:
        score_card.append(1)
    else:
        score_card.append(0)

# calculate score card
score_card_array = numpy.asfarray(score_card)
print(f"performance = : {score_card_array.sum() / score_card_array.size}")

# save the trained model
import os
import pickle

models_folder = "models"
model_filename = "neural_net_number_recogniser_modelv1.pkl"
model_path = os.path.join(models_folder, model_filename)
with open(model_path, 'wb') as file:
    pickle.dump(neural_net, file)



