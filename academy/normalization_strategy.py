from abc import ABC, abstractmethod


class NormalizationStrategy(ABC):
    @abstractmethod
    def normalize(self, data):
        pass
