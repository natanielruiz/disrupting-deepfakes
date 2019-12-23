import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=1, a=0.05, feat = None):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device
        self.feat = feat

    def perturb(self, X_nat, y, c_trg):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X = X_nat.clone().detach_()

        for i in range(self.k):
            # print(i)
            X.requires_grad = True
            output, feats = self.model(X, c_trg)

            if self.feat:
                print('self.feat ', self.feat)
                output = feats[self.feat]
                y = np.zeros(output.shape)
                y = torch.FloatTensor(y).to(self.device)

            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, eta

def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res