import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

import defenses.smoothing as smoothing

class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=30, a=0.01, feat = None):
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
                # print('self.feat ', self.feat)
                output = feats[self.feat]
                y = np.zeros(output.shape)
                y = torch.FloatTensor(y).to(self.device)

            self.model.zero_grad()
            loss = -self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat

    def perturb_blur(self, X_nat, y, c_trg):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X = X_nat.clone().detach_()
        X_orig = X_nat.clone().detach_()

        ks = 11
        sig = 1

        # preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks).to(self.device)
        preproc = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks).to(self.device)

        # blurred_image = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks).to(self.device)(X_orig)
        blurred_image = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks).to(self.device)(X_orig)

        for i in range(self.k):
            # print(i)
            X.requires_grad = True
            output, feats = self.model.forward_blur(X, c_trg, preproc)

            if self.feat:
                # print('self.feat ', self.feat)
                output = feats[self.feat]
                y = np.zeros(output.shape)
                y = torch.FloatTensor(y).to(self.device)

            self.model.zero_grad()
            loss = -self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        self.model.zero_grad()

        return X, X - X_nat, blurred_image

    def perturb_blur_iter(self, X_nat, y, c_trg):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X = X_nat.clone().detach_()

        ks_gauss = 11
        ks_avg = 3
        sig = 1
        blur_type = 1

        for i in range(self.k):
            # Declare smoothing layer
            if blur_type == 1:
                preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=ks_avg).to(self.device)
            elif blur_type == 2:
                preproc = smoothing.GaussianSmoothing2D(sigma=sig, channels=3, kernel_size=ks_gauss).to(self.device)

            X.requires_grad = True
            output, feats = self.model.forward_blur(X, c_trg, preproc)

            if self.feat:
                # print('self.feat ', self.feat)
                output = feats[self.feat]
                y = np.zeros(output.shape)
                y = torch.FloatTensor(y).to(self.device)

            self.model.zero_grad()
            loss = -self.loss_fn(output, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            if blur_type == 1:
                sig += 0.2
                if sig == 3.2:
                    blur_type = 2
                    sig = 1
            if blur_type == 2:
                ks_avg += 2
                if ks_avg == 11:
                    blur_type = 1
                    ks_avg = 3

            print(blur_type, sig, ks_avg)


        self.model.zero_grad()

        return X, X - X_nat

def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res