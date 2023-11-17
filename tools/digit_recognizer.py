import os
import pickle
import numpy
from PIL import Image
from academy.digit_normalization_strategy import DigitNormalizationStrategy


class DigitRecognizer:
    def __init__(self):
        self.normalization_strategy = DigitNormalizationStrategy()
        self.model = self.load_model()

    def load_model(self):
        model_path = input("please enter the path to the model: or leave it blank to use default model: ")
        if not model_path:
            models_folder = "../models"
            model_filename = "digitrecognizer.pkl"
            model_path = os.path.join(models_folder, model_filename)
        with open(model_path, 'rb') as file:
            return pickle.load(file)

    def convert_image_to_csv(self) -> str:
        # Open the image file
        image_path = input("Image path: ")
        image = Image.open(image_path)

        # Convert the image to grayscale
        grayscale_image = image.convert('L')

        # Threshold the image to separate the digit from the background
        threshold = 200  # Adjust this threshold based on your image
        binary_image = grayscale_image.point(lambda p: 0 if p < threshold else 255)

        # Find bounding box containing the digit and crop to that region
        bbox = binary_image.getbbox()
        cropped_image = binary_image.crop(bbox)

        # Resize the cropped image to 28x28 pixels
        resized_image = cropped_image.resize((28, 28))

        # Flatten the image into a 1D array
        flattened_image = list(resized_image.getdata())

        # Normalize pixel values to be between 0 and 255
        normalized_values = [255 - x for x in flattened_image]  # Invert pixel values if needed

        # Convert values to a comma-separated string
        csv_string = ','.join(map(str, normalized_values))

        return csv_string

    def inference(self, inputs: []):
        normalized_inputs = self.normalization_strategy.normalize(inputs)
        outputs = self.model.query(normalized_inputs)
        inf = numpy.argmax(outputs)
        print(f"Inference:  {inf}")


if __name__ == "__main__":
    art_file_path = "../art/logo.txt"

    with open(art_file_path, "r") as art_file:
        neural_net_art = art_file.read()

    # Display ASCII art to the user
    print(neural_net_art)
    digit_recognizer = DigitRecognizer()
    inputs = digit_recognizer.convert_image_to_csv()
    inputs = inputs.split(",")

    digit_recognizer.inference(inputs)
