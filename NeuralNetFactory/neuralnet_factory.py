from NeuralNetFactory.hyper_parameters import HyperParameters
from NeuralNets.i_neuralnet import INeuralNet
from NeuralNets.neuralnet_3_layers import NeuralNet3Layers
from NeuralNets.neuralnet_multi_layers import NeuralNetMultiLayers


class NeuralNetFactory:

    def collect_hyperparameters(self) -> HyperParameters | None:
        input_nodes = 0
        hidden_layers = []
        output_nodes = 0
        learning_rate = 0.1
        number_of_layers = int(input("Please enter to the total number of layers you want for your neural network: "
                                     "(minimum is 3:) "))
        while number_of_layers < 3:
            print("Total number of layers has to 3 or higher:")
            user_input = (input("Please enter to the total number of layers you want for your neural network: "
                                "or input e to exit"))
            if user_input == "e":
                return
            else:
                number_of_layers = int(user_input)
        number_of_hidden_layers = number_of_layers - 2
        input_nodes = int(input("Number of input nodes: "))
        for h in range(number_of_hidden_layers):
            hidden_layers.append(int(input(f"number of nodes for hidden layer {h+1}: ")))
        output_nodes = int(input("Number of output nodes: "))

        hyper_parameters = HyperParameters(input_nodes, hidden_layers, output_nodes, learning_rate)
        return hyper_parameters

    def create_neuralnet(self, hyper_parameters: HyperParameters) -> INeuralNet:
        if len(hyper_parameters.hidden_layers) < 2:
            number_of_hidden_nodes = hyper_parameters.hidden_layers[0]
            neural_net = NeuralNet3Layers(hyper_parameters.input_nodes, number_of_hidden_nodes,
                                          hyper_parameters.output_nodes, hyper_parameters.learning_rate)
        else:
            neural_net = NeuralNetMultiLayers(hyper_parameters.input_nodes, hyper_parameters.hidden_layers,
                                              hyper_parameters.output_nodes, hyper_parameters.learning_rate)
        return neural_net
