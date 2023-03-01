import abc
import numpy as np
from copy import copy

class Layer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, X):
        pass

    @abc.abstractmethod
    def backward(self, grad):
        pass

class Dense(Layer):
    """Dense layer
    Linear transformation of the input data: z = XW + b

    Parameters
    ----------
    input_dim : ints
        Number of input features
    output_dim : int
        Number of output features. If None, then output_dim = input_dim
    weight_init : str, optional
        Weight initialization method, by default "random"
    bias_init : str, optional
        Bias initialization method, by default "zeros"
    """
    def __init__(self, in_feature, out_feature=None, weight_init="random", bias_init="zeros"):
        self.in_feature = in_feature
        if in_feature == None:
            self.in_feature = in_feature
        else:
            self.out_feature = out_feature
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.W = np.random.rand(in_feature, out_feature)
        self.b = np.zeros(out_feature)

    def initialize(self, optimizer, regularization=None):
        self.optimizer = copy(optimizer)
        self.regularization = regularization
        
    def forward(self, X):
        self.X = X
        self.output = np.dot(X, self.W) + self.b
        return self.output

    def backward(self, grad):
        self.dW = np.dot(self.X.T, grad)
        self.db = np.sum(grad, axis=0)
        grad = np.dot(grad, self.W.T)

        self.W, self.b = self.optimizer.update([self.W, self.b], [self.dW, self.db])
        self.W, self.b = self.regularization.update([self.W, self.b])
        return grad   

class RNN(Layer):
    def __init__(self, in_feature, out_feature=None, weight_init="random", bias_init="zeros"):
        self.in_feature = in_feature
        if in_feature == None:
            self.in_feature = in_feature
        else:
            self.out_feature = out_feature
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.W_h = np.random.rand(in_feature, out_feature)
        self.W_x = np.random.rand(in_feature, out_feature)
        self.W_y = np.random.rand(in_feature, out_feature)
        self.b_h = np.zeros(out_feature)
        self.b_x = np.zeros(out_feature)
    
    def initialize(self, optimizer, regularization=None):
        self.optimizer = copy(optimizer)
        self.regularization = regularization
    
    def _cell(self, xt, ht):
        ht_out = np.tanh(self.W_x.dot(xt)) + np.dot(self.W_h, ht)
        yt = np.dot(self.W_y, ht_out)
        return yt, ht_out
        
    def forward(self, X):
        self.h = np.zeros(self.W_h.T.shape)
        for i in range(len(self.h)):
            self.h[i+1] = self.W_h * self.h[i] + self.W_x * self.x[i]
        
        return self.h
        
    def backward(self, grad):
        return grad

class Conv2D(Layer):
    """2D Convolutional Layer
    
    Parameters
    ----------
    input_dim : tuple
    """
    def __init__(self, n_filters, filter_shape, input_shape=None, padding='same', stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.trainable = True

    def initialize(self, optimizer, regularization=None):
        self.optimizer = copy(optimizer)
        self.regularization = regularization
    
    def forward(self, X):
        batch_size, channels, height, width = X.shape
        self.input_layer = X
        
        # self.X_col = image_to_column(X, self.filter_shape, stride=self.stride, output_shape=self.padding)
        # Turn weights into column shape
        self.W_col = self.W.reshape((self.n_filters, -1))
        # Calculate output
        output = self.W_col.dot(self.X_col) + self.w0
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(self.output_shape() + (batch_size, ))
        # Redistribute axises so that batch size comes first
        return output.transpose(3,0,1,2)
    
class MaxPool2D(Layer):
    """2D Max Pooling Layer

    Parameters
    ----------
    input_dim : tuple
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
    
    def forward(self, X):
        self.X = X
        return self.output
    
    def backward(self, grad):
        m = self.X.shape[0]
        dw = np.dot(self.X.T, grad) / m
        db = np.sum(grad, axis=0, keepdims=True) / m
        if self.regularization is not None:
            dw = dw + self.regularization(self.W)
            db = db + self.regularization(self.b)

        dX = np.dot(grad, self.W.T)
        
        self.W = self.W - dw 
        self.X = self.X - db 
        return dX

class Flatten(Layer):
    """Flatten Layer

    Parameters
    ----------
    input_dim : tuple
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
    
    def forward(self, X):
        self.X = X
        return X.reshape((X.shape[0], -1))

    def backward(self, grad):
        return grad.reshape(self.X.shape)

class Dropout(Layer):
    """Dropout Layer

    Parameters
    ----------
    p : float
    """
    def __init__(self, p):
        self.p = p
    
    def forward(self, X):
        c = (1 - self.p)
        self._mask = np.random.uniform(size=X.shape) > self.p
        c = self._mask
        return X * c
    
    def backward(self, grad):
        return grad * self._mask
    

class BatchNormalization(Layer):
    """Batch Normalization Layer

    Parameters
    ----------
    input_dim : int

    """
    def __init__(self, input_dim, eps=1e-8):
        self.input_dim = input_dim
        self.eps = eps
        self.mean = None
        self.var = None
        self.running_mean = np.zeros(self.input_dim)
        self.running_var = np.zeros(self.input_dim)
        self.gamma = np.ones(self.input_dim)
        self.beta = np.zeros(self.input_dim)

    def initialize(self, optimizer, regularization=None):
        self.optimizer = copy(optimizer)
        self.regularization = regularization

    def forward(self, X, train=True):
        self.X = X
        if train:
            self.mean = np.mean(X, axis=0)
            self.var = np.var(X, axis=0)
            self.running_mean = 0.9 * self.running_mean + 0.1 * self.mean 
            self.running_var = 0.9 * self.running_var + 0.1 * self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var

        self.h = (X - self.mean) / np.sqrt(self.var + self.eps)

        self.output = self.gamma * self.h + self.beta
        return self.output

     
    def backward(self, grad):
        m = self.h.shape[0]

        self.dgamma = np.sum(self.h * grad, axis=0)
        self.dbeta = np.sum(grad, axis=0)
        
        dh = grad * self.gamma
        grad = dh / np.sqrt(self.var + self.eps) + (np.sum(dh * (self.X - self.mean), axis=0) * -2 * (self.X - self.mean)) / m / (self.var + self.eps) / np.sqrt(self.var + self.eps) 
        self.gamma, self.beta = self.optimizer.update([self.gamma, self.beta], [self.dgamma, self.dbeta])
        self.gamma, self.beta = self.regularization.update([self.gamma, self.beta])
        
        return grad