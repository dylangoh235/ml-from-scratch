import numpy as np

class Logistic:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layers = []

    def accuracy(self, x, t):
        y = self.predict(x)
        return np.sum(y==t) / float(x.shape[0])    
    
    def train(self, X, y, epochs, verbose=False):
        self.W = np.random.rand(X.shape[1], y.shape[1])
        self.b = np.zeros(y.shape[1]) 
        train_errors = []
        train_acc_list = []
        m = X.shape[0]
        for epoch in range(epochs):

            Z = np.dot(X, self.W) + self.b
            A = 1 / (1 + np.exp(-Z))

            dW = np.dot(X.T, A - y)
            db = np.sum(A - y, axis=0) 

            self.W = self.W - self.learning_rate * dW
            self.b = self.b - self.learning_rate * db

            train_acc = self.accuracy(X, y)

            loss = - np.sum(y * np.log(A + 1e-9))
            train_errors.append(loss)
            train_acc_list.append(train_acc)

            print(f'{epoch}:{epochs} | Train accuracy: {train_acc}')

        return train_errors
    
    def predict(self, X):
        Z = np.dot(X, self.W) + self.b
        A = 1 / (1 + np.exp(-Z))
        Y = np.where( A > 0.5, 1, 0 )
        return Y
    