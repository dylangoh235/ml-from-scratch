import numpy as np
from machine_learning.utils.loss import Loss, CrossEntropy
from machine_learning.utils.regularization import Regularization, L2
from machine_learning.utils.optimization import Optimizer, Adam

class Network:
    def __init__(self, batch_size=64, regularization:Regularization=L2(), optimizer:Optimizer=Adam(), loss:Loss=CrossEntropy()):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.regularization = regularization
        self.loss_f = loss
        self.reg_loss = 0
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer, self.regularization)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
            if hasattr(layer, "W"):
                self.reg_loss = self.regularization.loss([layer.W])
        return X

    def backward(self, y):
        grad = y
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train(self, X, y, epochs, verbose=False): 
        train_size = X.shape[0]

        train_errors = []
        batch_per_epoch = int(np.ceil(train_size / self.batch_size))
        for epoch in range(epochs):
            for i in range(batch_per_epoch):
                _X = X[i * self.batch_size: (i + 1) * self.batch_size]
                _y = y[i * self.batch_size: (i + 1) * self.batch_size]
                output = self.forward(_X)

                loss = self.loss_f.forward(_y, output)
                train_errors.append(loss + self.reg_loss)
                grad = self.loss_f.backward(_y, output)
                self.backward(grad)
                
            print(f'{epoch}:{epochs}', end='\r')
        return train_errors
    
    def predict(self, X):
        y_hat = self.forward(X)
        return y_hat

    


    