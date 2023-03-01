import numpy as np
from machine_learning.utils.loss import Loss, CrossEntropy
from machine_learning.utils.regularization import Regularization, L2
from machine_learning.utils.optimization import Optimizer, Adam

class MLP:
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

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim !=1 : t = np.argmax(t, axis=1)
        return np.sum(y==t) / float(x.shape[0])    

    def backward(self, y):
        grad = y
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train(self, X, y, epochs, verbose=False): 
        train_size = X.shape[0]

        train_errors = []
        train_acc_list = []
        batch_per_epoch = int(np.ceil(train_size / self.batch_size))
        for epoch in range(epochs):
            for i in range(batch_per_epoch):
                _X = X[i * self.batch_size: (i + 1) * self.batch_size]
                _y = y[i * self.batch_size: (i + 1) * self.batch_size]

                output = self.forward(_X)
                loss = self.loss_f.forward(_y, output)
                train_errors.append(loss + self.reg_loss)
                self.backward(_y)
                
            train_acc = self.accuracy(X, y)
            train_acc_list.append(train_acc)
            print(f'{epoch}:{epochs} | Train accuracy: {train_acc}', end='\r')
        return train_errors
    
    def predict(self, X):
        y_hat = self.forward(X)
        return y_hat
    
    def get_params(self):
        params = {}
        for layer in self.layers:
            if hasattr(layer, 'W'):
                params[layer.__class__.__name__ + '_W'] = layer.W
                params[layer.__class__.__name__ + '_b'] = layer.b
        return params

    def reset(self):
        for layer in self.layers:
            if hasattr(layer, 'reset'):
                layer.reset()
            
    def summary(self):
        print("\n------------------")
        for layer in self.layers:
            if hasattr(layer, 'W'):
                print(f'{layer.__class__.__name__}: {layer.W.shape[0]} -> {layer.W.shape[1]}, # params: {layer.W.size + layer.b.size}')
            elif hasattr(layer, 'Activation'):
                print(f'{layer.__class__.__name__}: {layer.activation.__class__.__name__}')
        print("------------------")
    


    