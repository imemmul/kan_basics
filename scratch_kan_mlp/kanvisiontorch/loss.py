import torch
import torch.nn.functional as F

class Loss:
    def __call__(self, y, y_train):
        # y: output of network (logits or probabilities)
        # y_train: ground truth (one-hot encoded)
        self.y = y
        self.y_train = y_train

        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(0)
        if self.y_train.dim() == 1:
            self.y_train = self.y_train.unsqueeze(0)
        
        self.get_loss()
        self.get_dloss_dy()
        return self.loss

    def get_dloss_dy(self):
        pass

class CrossEntropyLoss(Loss):
    def get_loss(self):
        # y_train is a 2D tensor of the corresponding one-hot encoded labels
        m = self.y.size(0)  # Number of samples
        p = F.softmax(self.y, dim=1)  # Apply softmax to get probabilities

        if self.y_train.dim() == 1:
            log_likelihood = -torch.log(p[range(m), self.y_train])
        else:
            log_likelihood = -torch.log(p[range(m), torch.argmax(self.y_train, dim=1)])
        
        self.loss = torch.sum(log_likelihood) / m

    def get_dloss_dy(self):
        m = self.y.size(0)
        p = F.softmax(self.y, dim=1)
        
        if self.y_train.dim() == 1:
            p[range(m), self.y_train] -= 1
        else:
            p[range(m), torch.argmax(self.y_train, dim=1)] -= 1
        
        self.dloss_dy = p / m