"""
This file contains the tests for evaluating the functions in nn.py
"""
import sklearn.datasets as datasets
import nn
import numpy as np
from optimizer import optimizer


def test_run():
    """
    Sample test run.
    :return: None
    """
    # test run for binary classification problem:
    np.random.seed(3)
    print('Running a binary classification test')

    # Generate sample binary classification data
    data = datasets.make_classification(n_samples=30000, n_features=26, n_classes=2)
    X = data[0].T
    Y = (data[1].reshape(30000, 1)).T

    # optimize using  ADAM
    net = nn.nn([26, 32, 32, 1], ['relu', 'relu', 'sigmoid'], epochs=30)
    net.cost_function = 'CrossEntropyLoss'
    print('net architecture :')
    print(net)
    optim = optimizer.AdamOptimizer
    optim(X, Y, net, alpha=0.07, iterations=90, lamb=0.05, print_at=900)
    # output = net.forward(X)
    # output = 1 * (output >= 0.5)
    # accuracy = np.sum(output == Y) / 30000
    # print('for Adam:\n accuracy = ', accuracy * 100)


if __name__ == "__main__":
    test_run()
