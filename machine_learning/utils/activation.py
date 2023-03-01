import abc
import numpy as np

class Activation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, X):
        pass

    @abc.abstractmethod
    def backward(self, X, grad):
        pass

class ReLU(Activation):
    def forward(self, X):
        self.mask = (X<=0)
        out = X.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, grad):
        grad[self.mask] = 0
        return grad

class Sigmoid(Activation):
    """Sigmoid function
    f(x) = 1 / (1 + exp(-x))
    f'(x) = f(x) * (1 - f(x))

    Parameters
    ----------
    X : np.ndarray
    """
    def forward(self, X):
        self.X = X
        return 1 / (1 + np.exp(-X))

    def backward(self,  grad):
        return grad * self.forward(self.X) * (1 - self.forward(self.X))

class Softmax(Activation):
    def forward(self, X):
        self.X = X
        exps = np.exp(X - np.max(X, axis=-1, keepdims=True))
        self.A = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.A
    
    def backward(self, grad):
        m = grad.shape[0]
        grad = (self.A - grad) / m
        return grad
    
class Tanh(Activation):
    """Hyperbolic tangent
    f(x) = tanh(x)
    f'(x) = 1 - tanh(x) ** 2

    Parameters
    ----------
    X : np.ndarray
    """
    def forward(self, X):
        self.X = X
        return np.tanh(X)

    def backward(self, grad):
        return grad * (1 - np.tanh(self.X) ** 2)
