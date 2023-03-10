import abc
import numpy as np
from machine_learning.utils.activation import Tanh
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
        if out_feature is None:
            self.out_feature = in_feature
        else:
            self.out_feature = out_feature 
        self.weight_init = weight_init
        self.bias_init = bias_init

    def initialize(self, optimizer, regularization=None):
        self.W = np.random.rand(self.in_feature, self.out_feature)
        self.b = np.zeros(self.out_feature)
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
    """RNN Layer

    Parameters
    ----------
    input_dim : ints
        Number of input features

    weight_init : str, optional
        Weight initialization method, by default "random"
    bias_init : str, optional
        Bias initialization method, by default "zeros"

    reference: https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
    """
    def __init__(self, input_dim, n_units=3, activation=Tanh,  weight_init="random", bias_init="zeros"):
        self.input_dim = input_dim # (batch_size, time_steps, features)
        self.activation = activation()
        self.n_units = n_units
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.truncate = 5
    
    def initialize(self, optimizer, regularization=None):
        _, features = self.input_dim
        self.Wh = np.random.rand(self.n_units, self.n_units)
        self.WX = np.random.rand(features, self.n_units)
        self.Wy = np.random.rand(self.n_units, features)
        self.bh, self.by = np.zeros(self.n_units), np.zeros(features)

        self.optimizer = copy(optimizer)
        self.regularization = regularization

    def forward(self, X):
        batch_size, time_steps, _ = X.shape
        self.X = X
        self.z = np.zeros((batch_size, time_steps, self.n_units))
        self.h = np.zeros((batch_size, time_steps+1, self.n_units))
        self.y = np.zeros_like(self.X)

        for t in range(time_steps):
            self.z[:, t] = np.dot(X[:, t], self.WX) + np.dot(self.h[:, t-1], self.Wh) + self.bh
            self.h[:, t] = self.activation.forward(self.z[:, t])

            self.y[:, t] = np.dot(self.h[:, t],self.Wy) + self.by
        return self.y
    
    def backward(self, grad):
        _, time_steps, _ = self.X.shape

        dWy = np.zeros_like(self.Wy)
        dWh = np.zeros_like(self.Wh)
        dWX = np.zeros_like(self.WX)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(grad)

        for t in reversed(range(time_steps)):
            dWy = dWy + np.dot(self.h[:, t].T, grad[:, t])
            # dby = dby + grad[:,t]
            d_h = np.dot(grad[:, t], self.Wy.T) * self.activation.backward(self.z[:, t])
            dh_next[:, t] = np.dot(d_h, self.WX.T)
            # dbh = dbh + d_h
            for h_t in reversed(np.arange(max(0, t - self.truncate), t+1)):
                dWh = dWh + np.dot(d_h.T, self.h[:, h_t-1])
                dWX = dWX + np.dot(d_h.T, self.X[:, h_t]).T
                d_h = np.dot(d_h, self.Wh) * self.activation.backward(self.h[:, h_t-1])

        self.Wh, self.WX, self.Wy, self.bh, self.by = self.optimizer.update([self.Wh, self.WX, self.Wy, self.bh, self.by], [dWh, dWX, dWy, dbh, dby])
        return dh_next

class Conv2D(Layer):
    """2D Convolutional Layer
    
    Parameters
    ----------
    input_dim : tuple
    features : int
    kernel_size : tuple
    """
    def __init__(self, input_dim, kernel_size, filters = 3, stride=1, padding=0):
        self.input_dim = input_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def initialize(self, optimizer, regularization=None):
        self.kernel = np.random.rand(self.filters, *self.kernel_size)
        self.b = np.random.rand(1, self.filters, 1, 1) 
        self.optimizer = copy(optimizer)
        self.regularization = regularization

    def forward(self, img):
        self.img = img
        batch_size, channels, height, width = img.shape
        k_height, k_width = self.kernel_size
        output_height = int(np.ceil((height + 2*self.padding - k_height) / self.stride) + 1)
        output_width = int(np.ceil((width + 2*self.padding - k_width) / self.stride) + 1)
        output = np.zeros((batch_size, self.filters, output_height, output_width))

        for f in range(self.filters):
            for i in range(output_height):
                for j in range(output_width):
                    y_begin, y_end = i*self.stride, i*self.stride + k_height
                    x_begin, x_end = j*self.stride, j*self.stride + k_width
                    output[:, f, i, j] = np.sum(img[:, :, y_begin:y_end, x_begin:x_end] * self.kernel[f], axis=(1, 2, 3))
            output[:, f, :, :] += self.b[0, f, 0, 0] 
        return output

    def backward(self, grad):
        k_height, k_width = self.kernel_size
        output_height, output_width   = grad.shape[2], grad.shape[3]

        grad_img = np.zeros(self.img.shape)
        grad_kernel = np.zeros(self.kernel.shape)
        grad_b = np.sum(grad, axis=(0,2,3), keepdims=True)

        for f in range(self.filters):
            for i in range(output_height):
                for j in range(output_width):
                    y_begin, y_end = i*self.stride, i*self.stride + k_height
                    x_begin, x_end = j*self.stride, j*self.stride + k_width
                    grad_img[:, :, y_begin:y_end, x_begin:x_end] += grad[:, f, i, j][:, np.newaxis, np.newaxis, np.newaxis] * self.kernel[f]
                    grad_kernel += np.sum(self.img[:, :, y_begin:y_end, x_begin:x_end] * grad[:, f, i, j][:, np.newaxis, np.newaxis, np.newaxis], axis=0)
        if self.padding > 0:
            grad_img = grad_img[:, :, self.padding:-self.padding, self.padding:-self.padding]

        self.kernel, self.b= self.optimizer.update([self.kernel, self.b], [grad_kernel, grad_b])
        return grad_img

class MaxPool2D(Layer):
    """2D Max Pooling Layer

    Parameters
    ----------
    input_dim : tuple
    """
    def __init__(self):
        pass
    
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
    def __init__(self):
        pass
    
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
    def __init__(self, p=0.3):
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
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean = None
        self.var = None

    def initialize(self, optimizer, regularization=None):
        self.running_mean = None
        self.running_var = None
        self.gamma = None
        self.beta = None

        self.optimizer = copy(optimizer)
        self.regularization = regularization

    def forward(self, X, train=True):
        self.X = X
        input_dim = X.shape
        if self.running_mean is None:
            self.running_mean = np.zeros(input_dim)
            self.running_var = np.zeros(input_dim)
            self.gamma = np.ones(input_dim)
            self.beta = np.zeros(input_dim)
        
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