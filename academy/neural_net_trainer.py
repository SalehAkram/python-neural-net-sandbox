import numpy
import os
import pickle

from neuralnets.neuralnet import NeuralNet


class NeuralNetTrainer:

    def __init__(self, neural_net:NeuralNet, normalization_strategy, train_data_filepath, test_data_filepath):
        self.neural_net = neural_net
        self.normalization_strategy = normalization_strategy
        self.training_data_list = self.load_data(train_data_filepath)
        self.test_data_list = self.load_data(test_data_filepath)

    def load_data(self, filepath):
        data_file = open(filepath, 'r')
        data_list = data_file.readlines()
        data_file.close()
        return data_list

    def train_neural_net(self, epochs, target_lower_bound: float, target_upper_bound: float):
        for e in range(epochs):
            for record in self.training_data_list:
                # as we process through each record of the inputs we split the record by ',' into all values
                all_values = record.split(",")
                # the first index of all_values is our actual target and the rest are the inputs, we normalize the
                # inputs using our normalization strategy
                inputs = self.normalization_strategy.normalize(all_values[1:])
                # for digit recognition from 0 to 9 we have 10 output nodes, initially we set them all to 0.01
                targets = numpy.zeros(self.neural_net.hyper_parameters("on")) + target_lower_bound
                # if the first record had 5,0,0... 5 is our target so target [5] should be lit up
                # which means the all the other 9 output nodes/targets will be dimmed 0.01 and target[5] will be 0.99
                targets[int(all_values[0])] = target_upper_bound
                self.neural_net.train(inputs, targets)

    def test_neural_net(self) -> float:
        score_card = []
        for record in self.test_data_list:
            all_values = record.split(",")
            correct_label = int(all_values[0])
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            outputs = self.neural_net.query(inputs)
            # outputs = neural_net_v2.query(inputs)
            label = numpy.argmax(outputs)
            if label == correct_label:
                score_card.append(1)
            else:
                score_card.append(0)

        score_card_array = numpy.asfarray(score_card)
        return score_card_array.sum() / score_card_array.size

    def export_neural_net(self, filename: str):
        models_folder = "models"
        model_filename = f"{filename}.pkl"
        model_path = os.path.join(models_folder, model_filename)
        with open(model_path, 'wb') as file:
            pickle.dump(self.neural_net, file)
