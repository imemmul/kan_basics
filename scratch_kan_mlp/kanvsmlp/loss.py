import numpy as np
from .act import softmax

class Loss:
    def __call__(self, y, y_train):
        # y: output of network (logits or probabilities)
        # y_train: ground truth (one-hot encoded)
        self.y = np.array(y)
        self.y_train = np.array(y_train)
        
        if self.y.ndim == 1:
            self.y = np.expand_dims(self.y, axis=0)
        if self.y_train.ndim == 1:
            self.y_train = np.expand_dims(self.y_train, axis=0)
        
        self.get_loss()
        self.get_dloss_dy()
        return self.loss

    def get_dloss_dy(self):
        pass

class CrossEntropyLoss(Loss):
    def get_loss(self):
        # and y_train is a 2D array of the corresponding one-hot encoded labels
        m = self.y.shape[0]  # Number of samples
        p = softmax(self.y)  # Apply softmax to get probabilities
        # print(f"p.shape: {p.shape}, y_train.shape: {self.y_train.shape}")
        
        if self.y_train.ndim == 1:
            log_likelihood = -np.log(p[range(m), self.y_train])
        else:
            log_likelihood = -np.log(p[range(m), np.argmax(self.y_train, axis=1)])
        
        self.loss = np.sum(log_likelihood) / m

    def get_dloss_dy(self):
        m = self.y.shape[0]
        p = softmax(self.y)
        
        if self.y_train.ndim == 1:
            p[range(m), self.y_train] -= 1
        else:
            p[range(m), np.argmax(self.y_train, axis=1)] -= 1
        
        self.dloss_dy = p / m


class SquaredLoss(Loss):

    def get_loss(self):
        # compute loss l(xin, y)
        self.loss = np.mean(np.power(self.y - self.y_train, 2))

    def get_dloss_dy(self):
        # compute gradient of loss wrt xin
        self.dloss_dy = 2 * (self.y - self.y_train) / self.n_in
