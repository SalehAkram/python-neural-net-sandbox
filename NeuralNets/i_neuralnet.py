from abc import ABC, abstractmethod
from NeuralNetFactory.hyper_parameters import HyperParameters


class INeuralNet(ABC):
    @abstractmethod
    def hyper_parameters(self):
        pass

    @abstractmethod
    def train(self, input_list, target_list):
        pass

    @abstractmethod
    def query(self, input_list):
        pass
