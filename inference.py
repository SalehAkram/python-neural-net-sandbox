import os
import pickle

# Define the path to the "models" folder
models_folder = "models"
model_filename = "neural_net_number_recogniser_modelv1.pkl"
model_path = os.path.join(models_folder, model_filename)
with open(model_path, 'rb') as file:
    loaded_neuralNet = pickle.load(file)

# Example input data
input_data = [0.1, 0.2, 0.3, ...]  # Provide your input data

# Use the loaded model to make predictions
output = loaded_neuralNet.query(input_data)
print(output)
