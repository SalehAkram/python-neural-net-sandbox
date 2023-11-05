from neuralnet import NeuralNet
import numpy
import matplotlib.pyplot as plt
output_nodes = 10
input_nodes = 784
hidden_nodes = 100
learning_rate = 0.1
neuralNet = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

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
        neuralNet.train(inputs, targets)


# test the neural net
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
score_card = []

for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(all_values[0])
    #print(f"correct label: {correct_label}")
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = neuralNet.query(inputs)
    label = numpy.argmax(outputs)
    #print(f"networks answer: {label}")
    if label == correct_label:
        score_card.append(1)
    else:
        score_card.append(0)

# calculate score card
score_card_array = numpy.asfarray(score_card)
print(f"performance = : {score_card_array.sum()/score_card_array.size}")


# image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
# plt.imshow(image_array, cmap="Greys")
# plt.show()






