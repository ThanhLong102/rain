"""
Class containing functionality to build and train neural networks
Contains all activation and loss functions some other utility function.
"""
from typing import List, Tuple, Union

import numpy as np
from numpy import float64, ndarray


class nn:
    def __init__(self, layer_dimensions: List[int] = [], activations: List[str] = [], epochs=1) -> None:
        """
        Initializes network's weights and other useful variables.
        param layer_dimensions:
        :param activations: To store the activation for each layer
        -Parameters contains weights of the layer in form {'Wi':[],'bi':[]}
        -Cache contains intermediate results as [[A[i-1],Wi,bi],[Zi]], where i
         is layer number.
        -activations contains the names of activation function used for that layer
        -cost_function  contains the name of cost function to be used
        -grads contains the gradients calculated during back-prop in form {'dA(i-1)':[],'dWi':[],'dbi':[]}
        -layer_type contains the info about the type of layer( fc, etc)
        """
        self.parameters = {}
        self.cache = []
        self.activations = activations
        self.cost_function = ''
        self.cache_metrics = {'iteration': [], 'loss': [], 'accuracy': []}
        self.epochs = epochs
        self.grads = {}
        self.layer_type = ['']
        self.initialize_parameters(layer_dimensions)
        self.check_activations()

    def initialize_parameters(self, layer_dimensions: List[int]) -> None:
        """
        Saver initialization of weights of a network described by given layer
        dimensions.
        layer_dimensions: Dimensions to layers of the network
        :return: None
        """
        num_layers = int(len(self.parameters) / 2)

        for i in range(1, len(layer_dimensions)):
            self.parameters["W" + str(num_layers + i)] = (
                    np.sqrt(2 / layer_dimensions[i - 1]) * np.random.randn(layer_dimensions[i],
                                                                           layer_dimensions[i - 1])
            )
            self.parameters["b" + str(i + num_layers)] = np.zeros((layer_dimensions[i], 1))
            self.layer_type.append('fc')

    def check_activations(self) -> None:
        """
        Checks if activations for all layers are present. Adds 'None' if no activations are provided for a particular
        layer.

        :returns: None
        """
        num_layers = int(len(self.parameters) / 2)
        while len(self.activations) < num_layers:
            self.activations.append(None)

    @staticmethod
    def __linear_forward(A_prev, W, b):
        """
        Linear forward to the current layer using previous activations.
        A_prev: Previous Layer's activation
        W: Weights for current layer
        b: Biases for current layer
        :return: Linear cache and current calculated layer
        """
        Z = W.dot(A_prev) + b
        linear_cache = [A_prev, W, b]
        return Z, linear_cache

    def __activate(self, Z, n_layer=1):
        """
        Activate the given layer(Z) using the activation function specified by
        'type'.
        Note: This function treats 1 as starting index!
              First layer's index is 1.
        Z: Layer to activate
        n_layer: Layer's index
        :return: Activated layer and activation cache
        """

        act = None
        act_cache = [Z]
        if (self.activations[n_layer - 1]) is None:
            act = Z
        elif (self.activations[n_layer - 1]).lower() == "relu":
            act = Z * (Z > 0)
        elif (self.activations[n_layer - 1]).lower() == "tanh":
            act = np.tanh(Z)
        elif (self.activations[n_layer - 1]).lower() == "sigmoid":
            act = 1 / (1 + np.exp(-Z))
        elif (self.activations[n_layer - 1]).lower() == "softmax":
            act = np.exp(Z - np.max(Z))
            act = act / (act.sum(axis=0) + 1e-10)

        return act, act_cache

    def forward(self, net_input: ndarray) -> ndarray:
        """
        To forward propagate the entire Network.
        net_input: Contains the input to the Network
        :return: Output of the network
        """
        self.cache = []
        A = net_input
        for i in range(1, int(len(self.parameters) / 2 + 1)):
            W = self.parameters["W" + str(i)]
            b = self.parameters["b" + str(i)]
            Z = linear_cache = None
            if self.layer_type[i] == 'fc':
                Z, linear_cache = self.__linear_forward(A, W, b)

            # flatten the output if the next layer is fully connected
            A, act_cache = self.__activate(Z, i)
            self.cache.append([linear_cache, act_cache])

        # For Last Layer
        # W = self.parameters["W" + str(int(len(self.parameters) / 2))]
        # b = self.parameters["b" + str(int(len(self.parameters) / 2))]
        # Z, linear_cache = self.__linear_forward(A, W, b)
        # if len(self.activations) == len(self.parameters) / 2:
        #     A, act_cache = self.__activate(Z, len(self.activations))
        #     self.cache.append([linear_cache, act_cache])
        # else:
        #     A = Z
        #     self.cache.append([linear_cache, [None]])

        return A

    def MSELoss(self, prediction: ndarray, mappings: ndarray) -> float64:
        """
        Calculates the Mean Squared error with regularization cost(if provided) between output of the network and
        the real mappings of a function.
        Changes cost_function to appropriate value
        :param prediction: Output of the neural net
        :param mappings: Real outputs of a function
        :return: Mean squared error b/w output and mappings
        """

        self.cost_function = 'MSELoss'
        loss = np.square(prediction - mappings).mean()
        return loss

    def CrossEntropyLoss(self, prediction: ndarray, mappings: ndarray) -> float64:
        """
        Calculates the cross entropy loss between output of the network and the real mappings of a function
        Changes cost_function to appropriate value
        :param prediction: Output of the neural net
        :param mappings: Real outputs of a function
        :return: Mean squared error b/w output and mappings
        """
        epsilon = 1e-8
        self.cost_function = 'CrossEntropyLoss'
        loss = -(1 / prediction.shape[1]) * np.sum(
            mappings * np.log(prediction + epsilon) + (1 - mappings) * np.log(1 - prediction + epsilon))
        return loss

    def output_backward(self, prediction: ndarray, mapping: ndarray) -> ndarray:
        """
        Calculates the derivative of the output layer(dA)
        :param prediction: Output of neural net
        :param mapping: Correct output of the function
        :param cost_type: Type of Cost function used
        :return: Derivative of output layer, dA
        """
        dA = None
        cost = self.cost_function
        if cost.lower() == 'crossentropyloss':
            dA = -(np.divide(mapping, prediction + 1e-10) - np.divide(1 - mapping, 1 - prediction + 1e-10))

        elif cost.lower() == 'mseloss':
            dA = (prediction - mapping)

        return dA

    def deactivate(self, dA: ndarray, n_layer: int) -> Union[ndarray, int]:
        """
        Calculates the derivate of dA by deactivating the layer
        :param dA: Activated derivative of the layer
        :n_layer: Layer number to be deactivated
        :return: deact=> derivative of activation
        """
        act_cache = self.cache[n_layer - 1][1]
        dZ = act_cache[0]
        deact = None
        if self.activations[n_layer - 1] is None:
            deact = 1
        elif (self.activations[n_layer - 1]).lower() == "relu":
            deact = 1 * (dZ > 0)
        elif (self.activations[n_layer - 1]).lower() == "tanh":
            deact = 1 - np.square(dA)
        elif (self.activations[n_layer - 1]).lower() == "sigmoid" or (
                self.activations[n_layer - 1]).lower() == 'softmax':
            s = 1 / (1 + np.exp(-dZ + 1e-10))
            deact = s * (1 - s)

        return deact

    def linear_backward(self, dA: ndarray, n_layer: int) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Calculates linear backward propragation for layer denoted by n_layer
        :param dA: Derivative of cost w.r.t this layer
        :param n_layer: layer number
        :example: dA[1]=W[2]TdZ[2]
                dZ[1]=W[2]TdZ[2].∗g[1]′(z[1])
                dW[1]=1/mdZ[1]A[0]T
                dB[1]=1/mΣdZ[1]

        :return : dZ,dW,db,dA_prev
        """
        batch_size = dA.shape[1]
        current_cache = self.cache[n_layer - 1]
        linear_cache = current_cache[0]
        A_prev, W, b = linear_cache

        dZ = dA * self.deactivate(dA, n_layer)
        dW = (1 / batch_size) * dZ.dot(A_prev.T)
        db = (1 / batch_size) * np.sum(dZ, keepdims=True, axis=1)
        dA_prev = W.T.dot(dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dW, db, dA_prev

    def backward(self, prediction: ndarray, mappings: ndarray) -> None:
        """
        Backward propagates through the network and stores useful calculations
        :param prediction: Output of neural net
        :param mapping: Correct output of the function
        :return : None
        """
        layer_num = len(self.cache)
        doutput = self.output_backward(prediction, mappings)
        self.grads['dW' + str(layer_num)], self.grads['db' + str(layer_num)], self.grads[
            'dA' + str(layer_num - 1)] = self.linear_backward(doutput, layer_num)
        temp = self.layer_type
        self.layer_type = self.layer_type[1:]

        for l in reversed(range(layer_num - 1)):
            dW, db, dA_prev = None, None, None
            if self.layer_type[l] == 'fc':
                dW, db, dA_prev = self.linear_backward(self.grads['dA' + str(l + 1)], l + 1)
            self.grads['dW' + str(l + 1)] = dW
            self.grads['db' + str(l + 1)] = db
            self.grads['dA' + str(l)] = dA_prev

        self.layer_type = temp

    def __str__(self) -> str:
        """
        :Return: the network architecture and connectivity
        """
        net_string = ""
        for params in range(int(len(self.parameters) / 2)):
            weight = self.parameters['W' + str(params + 1)]
            net_string = net_string + " -> Linear(" + str(weight.shape[1]) + " , " + str(weight.shape[0]) + ")"
            if self.activations[params] is not None:
                net_string = net_string + " -> " + self.activations[params]
        return net_string
