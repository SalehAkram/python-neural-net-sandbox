from academy.neural_net_trainer import NeuralNetTrainer
from academy.digit_normalization_strategy import DigitNormalizationStrategy
from neuralnet_factory.neuralnet_factory import NeuralNetFactory
from tools.digit_recognizer import DigitRecognizer

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
normalization_strategy = DigitNormalizationStrategy()

print("Neural Net Training")
print("-------------------")
train_data_filepath = input("input training data file path: ")
test_data_filepath = input("input test data filepath: ")

trainer = NeuralNetTrainer(neural_net, normalization_strategy, train_data_filepath, test_data_filepath)
trainer.train_neural_net(2, 0.01, 0.99)
score = trainer.test_neural_net()
print(f"Neural Net Score: {score}")
choice = input("provide file name if you would like to export the neural net of type e to exit: ")
if choice.lower() != "e":
    trainer.export_neural_net(choice)






