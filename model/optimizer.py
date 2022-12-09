"""
This class contains various optimizer and helper functions in one place for better and modular understanding of overall
library.
"""
import numpy as np


class optimizer:
    @staticmethod
    def gradientDescentOptimizer(input, mappings, net, alpha=0.001, iterations=100, print_at=5, prnt=True,
                                 update=True):
        """
        Performs gradient descent on the given network setting the default value of epoch and alpha if not provided otherwise
        :param iterations: Number of iterations
        :param input  : input for neural net
        :param mapping: Correct output of the function
        :param net    : nn.nn object which provides the network architecture
        :param alpha  : Learning rate
        :param print_at: Print at multiples of 'print_at'
        :param prnt   : Print if prnt=true
        """

        for i in range(iterations):
            net.cache = []
            prediction = net.forward(input)
            loss_function = net.cost_function.lower()
            loss, regularization_cost = None, 0
            if loss_function == 'mseloss':
                loss = net.MSELoss(prediction, mappings)
            if loss_function == 'crossentropyloss':
                loss = net.CrossEntropyLoss(prediction, mappings)

            if prnt and i % print_at == 0:
                print('Loss at ', i, ' ', loss)

            net.backward(prediction, mappings)
            if update:
                net.parameters = optimizer.update_params(net.parameters, net.grads, alpha)

    @staticmethod
    def AdamOptimizer(input, mappings, net, alpha=0.001, betas=(0.9, 0.999), batch_size=64, print_at=5,
                      iterations=1):
        """
        Performs Adam optimization on the given network.
        :param batch_size: Integer or `None`. Number of samples per batch of computation
        :param iterations: Number of iterations
        :param input  : input for neural net
        :param mapping: Correct output of the function
        :param net    : nn.nn object which provides the network architecture
        :param alpha  : Learning rate
        :param betas: Adam Hyper parameters
        :param print_at: Print at multiples of 'print_at'
        :param prnt   : Print if prnt=true
        :return : None
        """
        velocity, square = {}, {}
        list_iteration = []
        list_loss = []
        list_accuracy = []
        for i in range(int(len(net.parameters) / 2)):
            velocity['dW' + str(i + 1)] = np.zeros(net.parameters['W' + str(i + 1)].shape)
            velocity['db' + str(i + 1)] = np.zeros(net.parameters['b' + str(i + 1)].shape)
            square['dW' + str(i + 1)] = np.zeros(net.parameters['W' + str(i + 1)].shape)
            square['db' + str(i + 1)] = np.zeros(net.parameters['b' + str(i + 1)].shape)

        epochs = net.epochs - 1
        while epochs >= 0:
            print('Epochs ', net.epochs - epochs, '/', net.epochs)

            input = np.take(input, np.random.permutation(input.shape[0]), axis=0, out=input)
            mappings = np.take(mappings, np.random.permutation(mappings.shape[0]), axis=0, out=mappings)

            iterations = int(input.shape[1] / batch_size)

            for i in range(1, iterations + 1):
                X = input[:, (i - 1) * batch_size: i * batch_size]
                Y = mappings[:, (i - 1) * batch_size: i * batch_size]

                optimizer.gradientDescentOptimizer(X, Y, net, iterations=1, prnt=False, update=False)

                for j in range(int(len(net.parameters) / 2)):
                    velocity['dW' + str(j + 1)] = betas[0] * velocity['dW' + str(j + 1)] + (1 - betas[0]) * net.grads[
                        'dW' + str(j + 1)]
                    velocity['db' + str(j + 1)] = betas[0] * velocity['db' + str(j + 1)] + (1 - betas[0]) * net.grads[
                        'db' + str(j + 1)]
                    square['dW' + str(j + 1)] = betas[1] * square['dW' + str(j + 1)] + (1 - betas[1]) * np.power(
                        net.grads['dW' + str(j + 1)], 2)
                    square['db' + str(j + 1)] = betas[1] * square['db' + str(j + 1)] + (1 - betas[1]) * np.power(
                        net.grads['db' + str(j + 1)], 2)

                update = {}
                for j in range(int(len(net.parameters) / 2)):
                    update['dW' + str(j + 1)] = velocity['dW' + str(j + 1)] / (
                            np.sqrt(square['dW' + str(j + 1)]) + 1e-10)
                    update['db' + str(j + 1)] = velocity['db' + str(j + 1)] / (
                            np.sqrt(square['db' + str(j + 1)]) + 1e-10)

                net.parameters = optimizer.update_params(net.parameters, update, alpha)

                prediction = net.forward(X)
                loss = None
                loss_function = net.cost_function.lower()
                if loss_function == 'mseloss':
                    loss = net.MSELoss(prediction, Y)
                if loss_function == 'crossentropyloss':
                    loss = net.CrossEntropyLoss(prediction, Y)
                output = 1 * (prediction >= 0.5)
                accuracy = np.sum(output == Y) / batch_size
                if i % print_at == 0:
                    print('At:', i, '[==========>] Loss', loss, ' - accuracy:', accuracy)

                # Add metrics
                list_iteration.append(i)
                list_loss.append(loss)
                list_accuracy.append(accuracy)
            net.cache_metrics = {'iteration': list_iteration,
                                 'loss': list_loss,
                                 'accuracy': list_accuracy}
            epochs = epochs - 1


    @staticmethod
    def update_params(params, updation, learning_rate):
        """
        Updates the parameters using gradients and learning rate provided

        :param params: Parameters of the network
        :param updation: updation valcues calculated using appropriate algorithms
        :param learning_rate: Learning rate for the updation of values in params
        :return : Updated params
        """

        for i in range(int(len(params) / 2)):
            params['W' + str(i + 1)] = params['W' + str(i + 1)] - learning_rate * updation['dW' + str(i + 1)]
            params['b' + str(i + 1)] = params['b' + str(i + 1)] - learning_rate * updation['db' + str(i + 1)]

        return params
