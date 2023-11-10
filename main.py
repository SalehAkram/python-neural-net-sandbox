from NeuralNetFactory.neuralnet_factory import NeuralNetFactory
import numpy
import matplotlib.pyplot as plt

# Read ASCII art from file
art_file_path = "art/logo.txt"

with open(art_file_path, "r") as art_file:
    neural_net_art = art_file.read()

# Display ASCII art to the user
print(neural_net_art)
# digit recognizer 3 layers 784, 100, 10

neuralnet_factory = NeuralNetFactory()
hyper_parameters = neuralnet_factory.collect_hyperparameters()
neural_net = neuralnet_factory.create_neuralnet(hyper_parameters)

# # train the neural net
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 2
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(",")
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(hyper_parameters.output_nodes) + 0.01
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

# # save the trained model
# import os
# import pickle
#
# models_folder = "models"
# model_filename = "neural_net_number_recogniser_modelv1.pkl"
# model_path = os.path.join(models_folder, model_filename)
# with open(model_path, 'wb') as file:
#     pickle.dump(neural_net, file)
