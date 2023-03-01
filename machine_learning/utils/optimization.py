import abc
import numpy as np

class Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, params, grad):
        pass

class SGD(Optimizer):
    """Stochastic Gradient Descent
    The SGD optimizer updates the parameters by introducting a momentum term.
    We can think of the momentum term in terms of Physics. There are two forces acting on a particle.
    One force, proportional to negative gradient of the cost function, is particle moving along the cost function surface. 
    And the other force, proportional to the velocity(derivative of theta), is the friction force. This prevents the particle from moving too fast.
    
    Parameters:
    -----------
    learning_rate: float
    momentum: float
    """
    def __init__(self, learning_rate:float=0.01, momentum:float=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:   
            self.velocity = [np.zeros_like(param) for param in params]

        for i, param in enumerate(params):
            change = self.momentum * self.velocity[i] + (1 - self.momentum) * grads[i]
            param = param - self.learning_rate * change
            params[i] = param
        
        return params

class SGDWithNesterov(Optimizer):
    """Stochastic Gradient Descent with Nesterov Momentum
    The SGD with Nesterov Momentum optimizer is similar to the SGD optimizer, but it  differs in the way the gradient is calculated.
    The gradient is calculated after the current velocity is evaluated. We can think of it as attempting to add a correction term to the velocity.
    In the convex batch gradient case, the rate of convergence is faster than SGD(1/k to 1/k^2), but in stochastic gradient case does not improve.

    Parameters:
    -----------
    learning_rate: float
    momentum: float
    """
    def __init__(self, learning_rate:float, momentum:float=0.01):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, grad_f):
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]


        # Doen't work for now
        for i, param in enumerate(params):
            params_tilde = param + self.momentum * self.velocity[i]
            grad = grad_f(params_tilde)
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad[i]
            param = param + self.velocity[i]
            params[i] = param
        
        return params

class AdaGrad(Optimizer):
    """AdaGrad
    The AdaGrad optimizer is individually adapts the learning rate of each parameter by scaling them inversely proportional to the squared root of the sum of their past squared gradients.
    The parameter with largest partial derivative will have the smallest learning rate, and vice versa. 
    AdaGrad performs well for some but not all since it can result in a premature and excessive decrease in the learning rate.

    Parameters:
    -----------
    learning_rate: float
    epsilon: float
    """
    def __init__(self, learning_rate:float, epsilon:float=1e-7):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.r = None

    def update(self, params, grads):
        if self.r is None:
            self.r = [np.zeros_like(param) for param in params]

        for i, param in enumerate(params):
            self.r[i] = self.r[i] + np.square(grads[i])
            param = param - self.learning_rate * grads[i] / (np.sqrt(self.r[i]) + self.epsilon)
            params[i] = param
        return params   
    
class RMSProp(Optimizer):
    """RMSProp
    The RMSProp optimizer is similar to the AdaGrad optimizer, but it utilizes a moving average of the squared gradients.
    The decay rate is used to discard the extreme past gradients so it can converge rapidly after finding a convex bowl.
    It performs better in the non-convex setting than AdaGrad.

    Parameters:
    -----------
    learning_rate: float
    momentum: float
        It is nesterov momentum.
    decay_rate: float
    epsilon: float
    """
    def __init__(self, learning_rate, decay_rate=0.9, momentum=0.0, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.r = None
    
    def update(self, params, grads):
        if self.r is None:
            self.r = [np.zeros_like(param) for param in params]

        for i, param in enumerate(params):
            self.r[i] = self.decay_rate * self.r[i] + (1 - self.decay_rate) * np.square(grads[i])
            param = param - self.learning_rate * grads[i] / (np.sqrt(self.r[i] + self.epsilon))
            params[i] = param

        return params

class RMSPropWithNesterov(Optimizer):
    """RMSProp
    The RMSProp optimizer is similar to the AdaGrad optimizer, but it utilizes a moving average of the squared gradients.
    The decay rate is used to discard the extreme past gradients so it can converge rapidly after finding a convex bowl.
    It performs better in the non-convex setting than AdaGrad.

    Parameters:
    -----------
    learning_rate: float
    momentum: float
        It is nesterov momentum.
    decay_rate: float
    epsilon: float
    """
    def __init__(self, learning_rate, decay_rate=0.9, momentum=0.0, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.r = None
    
    def update(self, params, grads):
        if self.r is None:
            self.r = [np.zeros_like(param) for param in params]

        for i, param in enumerate(params):
            self.r[i] = self.decay_rate * self.r[i] + (1 - self.decay_rate) * np.square(grads[i])
            param = param - self.learning_rate * grads[i] / (np.sqrt(self.r[i] + self.epsilon))
            params[i] = param

        return params

class Adam(Optimizer):
    """Adam
    The Adam optimizer is a combination of the momentum term and RMSProp.
    It is generally regarded as being robust to hyperparameter tuning, though the learning rate occasionally needs to be tuned.

    Parameters:
    -----------
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.s = None
        self.r = None

    def update(self, params, grads):
        if self.s is None:
            self.s = [np.zeros_like(param) for param in params]
            self.r = [np.zeros_like(param) for param in params]

        for i, param in enumerate(params):
            self.s[i] = self.beta1 * self.s[i] + (1 - self.beta1) * grads[i]
            self.r[i] = self.beta2 * self.r[i] + (1 - self.beta2) * np.square(grads[i])
            self.t = self.t + 1

            s_hat = self.s[i] / (1 - np.power(self.beta1, self.t))
            r_hat = self.r[i] / (1 - np.power(self.beta2, self.t))
            param = param - self.learning_rate * s_hat / (np.sqrt(r_hat) + self.epsilon)
            params[i] = param
        
        return params


# From here, the following optimizers are second order methods.
class NewtonMethod(Optimizer):
    """Newton's Method
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, params, grad):
        pass

class BFGS(Optimizer):
    """BFGS
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, params, grad):
        pass
        