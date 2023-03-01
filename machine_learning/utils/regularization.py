import abc
import numpy as np

class Regularization(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def loss(self, W):
        pass

    @abc.abstractmethod
    def update(self, W):
        pass

class L1(Regularization):
    """L1 Parameter Regularization.

    Parameters
    ----------
    alpha : float
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def loss(self, params):
        for i, param in enumerate(params):
            param = self.alpha * np.sum(np.abs(param))
            params[i] = param
        return np.sum(params)

    def update(self, params):
        for i, param in enumerate(params):
            param = self.alpha * np.sign(param)
            params[i] = params[i] + param
        return params

class L2(Regularization):
    """L2 Parameter Regularization also known as Ridge regression or Tikhonov regularization is a strategy to 
    drives the weights closer to near specific point in space (commonly zero). This is done by adding a regularization term
    to the loss function.

    Parameters
    ----------
    alpha : float
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def loss(self, params):
        for i, param in enumerate(params):
            param = self.alpha * np.sum(param**2)
            params[i] = param
        return np.sum(params)

    def update(self, params):
        for i, param in enumerate(params):
            param = self.alpha * param
            params[i] = params[i] + param
        return params

class MaxNorm(Regularization):
    """MaxNorm Parameter Regularization.

    Parameters
    ----------
    alpha : float
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    

