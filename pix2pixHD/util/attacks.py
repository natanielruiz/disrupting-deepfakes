import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=0.05, k=10, a=0.01):
        """
        FGSM, I-FGSM and PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss()

        # PGD or I-FGSM?
        self.rand = True

    def perturb(self, X_nat, y):
        """
        Vanilla Attack.
        """
        if self.rand:
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, X_nat.shape).astype('float32')).cuda()
        else:
            X = X_nat.clone().detach_()
            # use the following if FGSM or I-FGSM and random seeds are fixed
            X = X_nat.clone().detach_() + torch.tensor(np.random.uniform(-0.001, 0.001, X_nat.shape).astype('float32')).cuda() 

        for i in range(self.k):
            X.requires_grad = True
            output = self.model(X)

            self.model.zero_grad()

            # Minus in the loss means "towards" and plus means "away from"
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            eta, X_adv, loss, grad, output = None, None, None, None, None

        return X, X - X_nat

def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res