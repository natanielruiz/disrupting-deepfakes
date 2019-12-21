import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model import Generator
from model import Discriminator

import os
import re


class Utils:

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim,
                           self.g_repeat_num).to(self.device)
        self.D = Discriminator(
            self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num).to(self.device)

        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # TODO: implement data parallelization for multiple gpus
        # self.gpu_ids = torch.cuda.device_count()
        # print("GPUS AVAILABLE: ", self.gpu_ids)
        # if self.gpu_ids > 1:
        #     torch.nn.DataParallel(self.D, device_ids=list(range(self.gpu_ids)))
        #     torch.nn.DataParallel(self.G, device_ids=list(range(self.gpu_ids)))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)
        self.writer = SummaryWriter(logdir=self.log_dir)

    def smooth_loss(self, att):
        return torch.mean(torch.mean(torch.abs(att[:, :, :, :-1] - att[:, :, :, 1:])) +
                          torch.mean(torch.abs(att[:, :, :-1, :] - att[:, :, 1:, :])))

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def imFromAttReg(self, att, reg, x_real):
        """Mixes attention, color and real images"""
        return (1-att)*reg + att*x_real

    def create_labels(self, data_iter):
        """Return samples for visualization"""
        x, c = [], []
        x_data, c_data = data_iter.next()

        for i in range(self.num_sample_targets):
            x.append(x_data[i].repeat(
                self.batch_size, 1, 1, 1).to(self.device))
            c.append(c_data[i].repeat(self.batch_size, 1).to(self.device))

        return x, c

    def save_models(self, iteration, epoch):
        try:  # To avoid crashing on the first step
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-G.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-D.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-G_optim.ckpt'.format(iteration+1-self.model_save_step, epoch)))
            os.remove(os.path.join(self.model_save_dir,
                                   '{}-{}-D_optim.ckpt'.format(iteration+1-self.model_save_step, epoch)))
        except:
            pass

        G_path = os.path.join(self.model_save_dir,
                              '{}-{}-G.ckpt'.format(iteration+1, epoch))
        D_path = os.path.join(self.model_save_dir,
                              '{}-{}-D.ckpt'.format(iteration+1, epoch))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)

        G_path_optim = os.path.join(
            self.model_save_dir, '{}-{}-G_optim.ckpt'.format(iteration+1, epoch))
        D_path_optim = os.path.join(
            self.model_save_dir, '{}-{}-D_optim.ckpt'.format(iteration+1, epoch))
        torch.save(self.g_optimizer.state_dict(), G_path_optim)
        torch.save(self.d_optimizer.state_dict(), D_path_optim)

        print(f'Saved model checkpoints in {self.model_save_dir}...')

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}-{}...'.format(resume_iters, self.first_epoch))
        G_path = os.path.join(
            self.model_save_dir, '{}-{}-G.ckpt'.format(resume_iters, self.first_epoch))
        D_path = os.path.join(
            self.model_save_dir, '{}-{}-D.ckpt'.format(resume_iters, self.first_epoch))
        self.G.load_state_dict(torch.load(
            G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(
            D_path, map_location=lambda storage, loc: storage))

        G_optim_path = os.path.join(
            self.model_save_dir, '{}-{}-G_optim.ckpt'.format(resume_iters, self.first_epoch))
        D_optim_path = os.path.join(
            self.model_save_dir, '{}-{}-D_optim.ckpt'.format(resume_iters, self.first_epoch))
        self.d_optimizer.load_state_dict(torch.load(D_optim_path))
        self.g_optimizer.load_state_dict(torch.load(G_optim_path))

    def numericalSort(self, value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
