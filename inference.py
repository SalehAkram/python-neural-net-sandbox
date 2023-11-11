import os
import pickle
import numpy
# Define the path to the "models" folder
models_folder = "models"
model_filename = "digitrecognizer.pkl"
model_path = os.path.join(models_folder, model_filename)
with open(model_path, 'rb') as file:
    loaded_neuralNet = pickle.load(file)

# Example input data
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
all_values = test_data_list[6].split(",")
correct_label = int(all_values[0])
print(f"actual: {correct_label}")
inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

# Use the loaded model to make predictions
outputs = loaded_neuralNet.query(inputs)
predicted = numpy.argmax(outputs)
print(f"inference/prediction: {predicted}")
