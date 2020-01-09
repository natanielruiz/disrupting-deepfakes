import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

class LinfPGDAttack(object):
    def __init__(self, model=None, device=None, epsilon=0.05, k=20, a=0.01):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.loss_fn = nn.MSELoss().to(device)
        self.device = device

    def perturb(self, X_nat, y, c_trg):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X = X_nat.clone().detach_()

        for i in range(self.k):
            # print(i)
            X.requires_grad = True
            output_att, output_img = self.model(X, c_trg)

            out = imFromAttReg(output_att, output_img, X)

            self.model.zero_grad()
            # loss = -self.loss_fn(output_att, y) + self.loss_fn(output_img, y)
            loss = -self.loss_fn(out, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        return X, eta

    def perturb_iter_data(self, X_nat, X_all, y, c_trg):
        """
        X_nat is a tensor with several different images.
        This does not work at all yet..
        """
        X = X_nat.clone().detach_()
        # X_all_local = X_all.clone().detach_()

        j = 0
        J = X_all.size(0)
        J = 1

        for i in range(self.k):
            # print(i,j)
            X_j = X_all[j].unsqueeze(0)
            X_j.requires_grad = True
            output_att, output_img = self.model(X_j, c_trg)

            out = imFromAttReg(output_att, output_img, X_j)

            self.model.zero_grad()
            loss = -self.loss_fn(out, y)
            loss.backward()
            grad = X_j.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            j += 1
            if j == J:
                j = 0

        return X, eta

    def perturb_iter_class(self, X_nat, y, c_trg):
        """
        Iterative Class Conditional Attack
        """
        X = X_nat.clone().detach_()

        j = 0
        J = c_trg.size(0)

        for i in range(self.k):
            # print(i)
            X.requires_grad = True
            output_att, output_img = self.model(X, c_trg[j,:].unsqueeze(0))

            out = imFromAttReg(output_att, output_img, X)

            self.model.zero_grad()

            # Away from black
            loss = self.loss_fn(output_att, y)
            loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

            j += 1
            if j == J:
                j = 0

        return X, eta

    def perturb_joint_class(self, X_nat, y, c_trg):
        """
        Joint Class Conditional Attack
        """
        X = X_nat.clone().detach_()

        J = c_trg.size(0)
        full_loss = 0.0

        for i in range(self.k):
            for j in range(J):
                # print(i)
                X.requires_grad = True
                output_att, output_img = self.model(X, c_trg[j,:].unsqueeze(0))

                out = imFromAttReg(output_att, output_img, X)

                self.model.zero_grad()

                loss = -self.loss_fn(output_att, y)
                full_loss += loss

            full_loss.backward()
            grad = X.grad

            X_adv = X + self.a * grad.sign()

            eta = torch.clamp(X_adv - X_nat, min=-self.epsilon, max=self.epsilon)
            X = torch.clamp(X_nat + eta, min=-1, max=1).detach_()

        return X, eta

def clip_tensor(X, Y, Z):
    # Clip X with Y min and Z max
    X_np = X.data.cpu().numpy()
    Y_np = Y.data.cpu().numpy()
    Z_np = Z.data.cpu().numpy()
    X_clipped = np.clip(X_np, Y_np, Z_np)
    X_res = torch.FloatTensor(X_clipped)
    return X_res

def imFromAttReg(att, reg, x_real):
    """Mixes attention, color and real images"""
    return (1-att)*reg + att*x_real