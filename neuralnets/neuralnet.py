from abc import ABC, abstractmethod


class NeuralNet(ABC):
    @abstractmethod
    def hyper_parameters(self, key: str):
        pass

    @abstractmethod
    def train(self, input_list, target_list):
        pass

    @abstractmethod
    def query(self, input_list):
        pass
