import numpy
from academy.normalization_strategy import NormalizationStrategy


class DigitNormalizationStrategy(NormalizationStrategy):
    def normalize(self, data):
        # Implement normalization for digit recognition (0-9)
        normalized_data = (numpy.asfarray(data) / 255.0 * 0.99) + 0.01
        return normalized_data