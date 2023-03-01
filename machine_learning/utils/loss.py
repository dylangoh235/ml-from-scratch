import abc
import numpy as np

class Loss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, y, y_hat):
        pass

    @abc.abstractmethod
    def backward(self, y, y_hat):
        pass

class MSE(Loss):
    def forward(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)
    
    def backward(self, y, y_hat):
        return (y_hat - y) / y.size

class CrossEntropy(Loss):
    def forward(self, y, y_hat):
        return -np.sum(y * np.log(y_hat + 1e-9)) / y.shape[0]

    def backward(self, y, y_hat):
        grad = - (y / y_hat)
        return grad