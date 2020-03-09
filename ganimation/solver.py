import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from model import Generator, Discriminator
from utils import Utils

import numpy as np

import os
import time
import datetime
import random
import glob

import attacks


class Solver(Utils):

    def __init__(self, data_loader, config_dict):
        # NOTE: the following line create new class arguments with the
        # values in config_dict
        self.__dict__.update(**config_dict)

        self.data_loader = data_loader

        self.device = 'cuda:' + \
            str(self.gpu_id) if torch.cuda.is_available() else 'cpu'
        print(f"Model running on {self.device}")

        if self.use_tensorboard:
            self.build_tensorboard()

        self.loss_visualization = {}

        self.build_model()

    def train(self):
        print('Training...')

        self.global_counter = 0

        if self.resume_iters:
            self.first_iteration = self.resume_iters
            self.restore_model(self.resume_iters)
        else:
            self.first_iteration = 0

        self.start_time = time.time()

        for epoch in range(self.first_epoch, self.num_epochs):
            print(f"EPOCH {epoch} WITH {len(self.data_loader)} STEPS")

            self.alpha_rec = 1
            self.epoch = epoch

            for iteration in range(self.first_iteration, len(self.data_loader)):
                self.iteration = iteration
                self.get_training_data()
                self.train_discriminator()

                if (self.iteration+1) % self.n_critic == 0:
                    generation_outputs = self.train_generator()

                if (self.iteration+1) % self.sample_step == 0:
                    self.print_generations(generation_outputs)

                if self.iteration % self.model_save_step == 0:
                    self.save_models(self.iteration, self.epoch)

                if self.iteration % self.log_step == 0:
                    self.update_tensorboard()
                self.global_counter += 1

            # Decay learning rates.
            if (self.epoch+1) > self.num_epochs_decay:
                # float(self.num_epochs_decay))
                self.g_lr -= (self.g_lr / 10.0)
                # float(self.num_epochs_decay))
                self.d_lr -= (self.d_lr / 10.0)
                self.update_lr(self.g_lr, self.d_lr)
                print('Decayed learning rates, self.g_lr: {}, self.d_lr: {}.'.format(
                    self.g_lr, self.d_lr))

            # Save the last model
            self.save_models()

            self.first_iteration = 0  # Next epochs start from 0

    def get_training_data(self):
        try:
            self.x_real, self.label_org = next(self.data_iter)
        except:
            self.data_iter = iter(self.data_loader)
            self.x_real, self.label_org = next(self.data_iter)

        self.x_real = self.x_real.to(self.device)  # Input images.
        # Labels for computing classification loss.
        self.label_org = self.label_org.to(self.device)

        # Get random targets for training
        self.label_trg = self.get_random_labels_list()
        self.label_trg = torch.FloatTensor(self.label_trg).clamp(0, 1)
        # Labels for computing classification loss.
        self.label_trg = self.label_trg.to(self.device)

        if self.use_virtual:
            self.label_trg_virtual = self.get_random_labels_list()
            self.label_trg_virtual = torch.FloatTensor(
                self.label_trg_virtual).clamp(0, 1)
            # Labels for computing classification loss.
            self.label_trg_virtual = self.label_trg_virtual.to(self.device)

            assert not torch.equal(
                self.label_trg_virtual, self.label_trg), "Target label and virtual label are the same"

    def get_random_labels_list(self):
        trg_list = []
        for _ in range(self.batch_size):
            random_num = random.randint(
                0, len(self.data_loader)*self.batch_size-1)
            # Select a random AU vector from the dataset
            trg_list_aux = self.data_loader.dataset[random_num][1]
            # Apply a variance of 0.1 to the vector
            trg_list.append(trg_list_aux.numpy() +
                            np.random.uniform(-0.1, 0.1, trg_list_aux.shape))
        return trg_list

    def train_discriminator(self):
        # Compute loss with real images.
        critic_output, classification_output = self.D(self.x_real)
        d_loss_critic_real = -torch.mean(critic_output)
        d_loss_classification = torch.nn.functional.mse_loss(
            classification_output, self.label_org)

        # Compute loss with fake images.
        attention_mask, color_regression = self.G(self.x_real, self.label_trg)
        x_fake = self.imFromAttReg(
            attention_mask, color_regression, self.x_real)
        critic_output, _ = self.D(x_fake.detach())
        d_loss_critic_fake = torch.mean(critic_output)

        # Compute loss for gradient penalty.
        alpha = torch.rand(self.x_real.size(0), 1, 1, 1).to(self.device)
        # Half of image info from fake and half from real
        x_hat = (alpha * self.x_real.data + (1 - alpha)
                 * x_fake.data).requires_grad_(True)
        critic_output, _ = self.D(x_hat)
        d_loss_gp = self.gradient_penalty(critic_output, x_hat)

        # Backward and optimize.
        d_loss = d_loss_critic_real + d_loss_critic_fake + self.lambda_cls * \
            d_loss_classification + self.lambda_gp * d_loss_gp

        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Logging.
        self.loss_visualization['D/loss'] = d_loss.item()
        self.loss_visualization['D/loss_real'] = d_loss_critic_real.item()
        self.loss_visualization['D/loss_fake'] = d_loss_critic_fake.item()
        self.loss_visualization['D/loss_cls'] = self.lambda_cls * \
            d_loss_classification.item()
        self.loss_visualization['D/loss_gp'] = self.lambda_gp * \
            d_loss_gp.item()

    def train_generator(self):
        # Original-to-target domain.
        attention_mask, color_regression = self.G(self.x_real, self.label_trg)
        x_fake = self.imFromAttReg(
            attention_mask, color_regression, self.x_real)

        critic_output, classification_output = self.D(x_fake)
        g_loss_fake = -torch.mean(critic_output)
        g_loss_cls = torch.nn.functional.mse_loss(
            classification_output, self.label_trg)

        # Target-to-original domain.
        if not self.use_virtual:
            reconstructed_attention_mask, reconstructed_color_regression = self.G(
                x_fake, self.label_org)
            x_rec = self.imFromAttReg(
                reconstructed_attention_mask, reconstructed_color_regression, x_fake)

        else:
            reconstructed_attention_mask, reconstructed_color_regression = self.G(
                x_fake, self.label_org)
            x_rec = self.imFromAttReg(
                reconstructed_attention_mask, reconstructed_color_regression, x_fake)

            reconstructed_attention_mask_2, reconstructed_color_regression_2 = self.G(
                x_fake, self.label_trg_virtual)
            x_fake_virtual = self.imFromAttReg(
                reconstructed_attention_mask_2, reconstructed_color_regression_2, x_fake)

            reconstructed_virtual_attention_mask, reconstructed_virtual_color_regression = self.G(
                x_fake_virtual, self.label_trg)
            x_rec_virtual = self.imFromAttReg(
                reconstructed_virtual_attention_mask, reconstructed_virtual_color_regression, x_fake_virtual.detach())

        # Compute losses
        g_loss_saturation_1 = attention_mask.mean()
        g_loss_smooth1 = self.smooth_loss(attention_mask)

        if not self.use_virtual:
            g_loss_rec = torch.nn.functional.l1_loss(self.x_real, x_rec)
            g_loss_saturation_2 = reconstructed_attention_mask.mean()
            g_loss_smooth2 = self.smooth_loss(reconstructed_attention_mask)

        else:
            g_loss_rec = (1-self.alpha_rec)*torch.nn.functional.l1_loss(self.x_real, x_rec) + \
                self.alpha_rec * \
                torch.nn.functional.l1_loss(x_fake, x_rec_virtual)

            g_loss_saturation_2 = (1-self.alpha_rec) * reconstructed_attention_mask.mean() + \
                self.alpha_rec * reconstructed_virtual_attention_mask.mean()

            g_loss_smooth2 = (1-self.alpha_rec) * self.smooth_loss(reconstructed_virtual_attention_mask) + \
                self.alpha_rec * self.smooth_loss(reconstructed_attention_mask)

        g_attention_loss = self.lambda_smooth * g_loss_smooth1 + self.lambda_smooth * g_loss_smooth2 \
            + self.lambda_sat * g_loss_saturation_1 + self.lambda_sat * g_loss_saturation_2

        g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + \
            self.lambda_cls * g_loss_cls + g_attention_loss

        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

        # Logging.
        self.loss_visualization['G/loss'] = g_loss.item()
        self.loss_visualization['G/loss_fake'] = g_loss_fake.item()
        self.loss_visualization['G/loss_rec'] = self.lambda_rec * \
            g_loss_rec.item()
        self.loss_visualization['G/loss_cls'] = self.lambda_cls * \
            g_loss_cls.item()
        self.loss_visualization['G/attention_loss'] = g_attention_loss.item()
        self.loss_visualization['G/loss_smooth1'] = self.lambda_smooth * \
            g_loss_smooth1.item()
        self.loss_visualization['G/loss_smooth2'] = self.lambda_smooth * \
            g_loss_smooth2.item()
        self.loss_visualization['G/loss_sat1'] = self.lambda_sat * \
            g_loss_saturation_1.item()
        self.loss_visualization['G/loss_sat2'] = self.lambda_sat * \
            g_loss_saturation_2.item()
        self.loss_visualization['G/alpha'] = self.alpha_rec

        if not self.use_virtual:
            return {
                "color_regression": color_regression,
                "x_fake": x_fake,
                "attention_mask": attention_mask,
                "x_rec": x_rec,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_color_regression": reconstructed_color_regression,
            }

        else:
            return {
                "color_regression": color_regression,
                "x_fake": x_fake,
                "attention_mask": attention_mask,
                "x_rec": x_rec,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_color_regression": reconstructed_color_regression,
                "reconstructed_virtual_attention_mask": reconstructed_virtual_attention_mask,
                "reconstructed_virtual_color_regression": reconstructed_virtual_color_regression,
                "x_rec_virtual": x_rec_virtual,
            }

    def print_generations(self, generator_outputs_dict):
        print_epoch_images = False
        save_image(self.denorm(self.x_real), self.sample_dir +
                   '/{}_4real_.png'.format(self.epoch))
        save_image((generator_outputs_dict["color_regression"]+1)/2,
                   self.sample_dir + '/{}_2reg_.png'.format(self.epoch))
        save_image(self.denorm(
            generator_outputs_dict["x_fake"]), self.sample_dir + '/{}_3res_.png'.format(self.epoch))
        save_image(generator_outputs_dict["attention_mask"],
                   self.sample_dir + '/{}_1attention_.png'.format(self.epoch))
        save_image(self.denorm(
            generator_outputs_dict["x_rec"]), self.sample_dir + '/{}_5rec_.png'.format(self.epoch))

        if not self.use_virtual:
            save_image(generator_outputs_dict["reconstructed_attention_mask"],
                       self.sample_dir + '/{}_6rec_attention.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["reconstructed_color_regression"]), self.sample_dir + '/{}_7rec_reg.png'.format(self.epoch))

        else:
            save_image(generator_outputs_dict["reconstructed_attention_mask"],
                       self.sample_dir + '/{}_6rec_attention_.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["reconstructed_color_regression"]), self.sample_dir + '/{}_7rec_reg.png'.format(self.epoch))

            save_image(generator_outputs_dict["reconstructed_virtual_attention_mask"],
                       self.sample_dir + '/{}_8rec_virtual_attention.png'.format(self.epoch))
            save_image(self.denorm(generator_outputs_dict["reconstructed_virtual_color_regression"]),
                       self.sample_dir + '/{}_91rec_virtual_reg.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["x_rec_virtual"]), self.sample_dir + '/{}_92rec_epoch_.png'.format(self.epoch))

    def update_tensorboard(self):
        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}],  [{}/{}], Epoch [{}/{}]".format(
            et, self.iteration+1, len(self.data_loader), self.epoch+1, self.num_epochs)
        for tag, value in self.loss_visualization.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

        if self.use_tensorboard:
            for tag, value in self.loss_visualization.items():
                self.writer.add_scalar(
                    tag, value, global_step=self.global_counter)

    def animation(self, mode='animate_image'):
        from PIL import Image
        from torchvision import transforms as T

        regular_image_transform = []
        regular_image_transform.append(T.ToTensor())
        regular_image_transform.append(T.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        regular_image_transform = T.Compose(regular_image_transform)

        G_path = sorted(glob.glob(os.path.join(
            self.animation_models_dir, '*G.ckpt')), key=self.numericalSort)[0]
        self.G.load_state_dict(torch.load(G_path, map_location=f'cuda:{self.gpu_id}'))
        self.G = self.G.cuda(0)

        reference_expression_images = []

        with torch.no_grad():
            with open(self.animation_attributes_path, 'r') as txt_file:
                csv_lines = txt_file.readlines()

                targets = torch.zeros(len(csv_lines), self.c_dim)
                input_images = torch.zeros(len(csv_lines), 3, 128, 128)

                for idx, line in enumerate(csv_lines):
                    splitted_lines = line.split(' ')
                    image_path = os.path.join(
                        self.animation_attribute_images_dir, splitted_lines[0])
                    input_images[idx, :] = regular_image_transform(
                        Image.open(image_path)).cuda()
                    reference_expression_images.append(splitted_lines[0])
                    targets[idx, :] = torch.Tensor(
                        np.array(list(map(lambda x: float(x)/5., splitted_lines[1::]))))

        if mode == 'animate_random_batch':
            animation_batch_size = 7

            self.data_iter = iter(self.data_loader)
            images_to_animate, _ = next(self.data_iter)
            images_to_animate = images_to_animate[0:animation_batch_size].cuda(
            )

            for target_idx in range(targets.size(0)):
                targets_au = targets[target_idx, :].unsqueeze(
                    0).repeat(animation_batch_size, 1).cuda()
                resulting_images_att, resulting_images_reg = self.G(
                    images_to_animate, targets_au)

                resulting_images = self.imFromAttReg(
                    resulting_images_att, resulting_images_reg, images_to_animate).cuda()

                save_images = - \
                    torch.ones((animation_batch_size + 1)
                               * 2, 3, 128, 128).cuda()

                save_images[1:animation_batch_size+1] = images_to_animate
                save_images[animation_batch_size+1] = input_images[target_idx]
                save_images[animation_batch_size +
                            2:(animation_batch_size + 1)*2] = resulting_images

                save_image((save_images+1)/2, os.path.join(self.animation_results_dir,
                                                           reference_expression_images[target_idx]))

        if mode == 'animate_image':
            # Initialize Metrics
            l1_error, l2_error, min_dist, l0_error = 0.0, 0.0, 0.0, 0.0
            n_dist, n_samples = 0, 0

            pgd_attack = attacks.LinfPGDAttack(model=self.G, device=self.device)

            images_to_animate_path = sorted(glob.glob(
                self.animation_images_dir + '/*'))

            for idx, image_path in enumerate(images_to_animate_path):
                image_to_animate = regular_image_transform(Image.open(image_path)).unsqueeze(0).cuda()

                for target_idx in range(targets.size(0)-1):
                    print('image', idx, 'AU', target_idx)

                    targets_au = targets[target_idx, :].unsqueeze(0).cuda()

                    with torch.no_grad():
                        resulting_images_att_noattack, resulting_images_reg_noattack = self.G(
                            image_to_animate, targets_au)
                        resulting_image_noattack = self.imFromAttReg(
                            resulting_images_att_noattack, resulting_images_reg_noattack, image_to_animate).cuda()

                    # Transfer to different classes
                    # if target_idx == 0:
                        # Wrong Class
                        # x_adv, perturb = pgd_attack.perturb(image_to_animate, image_to_animate, targets[0, :].unsqueeze(0).cuda())

                        # Joint Class Conditional
                        # x_adv, perturb = pgd_attack.perturb_joint_class(image_to_animate, image_to_animate, targets[:, :].cuda())

                        # Iterative Class Conditional
                        # x_adv, perturb = pgd_attack.perturb_iter_class(image_to_animate, image_to_animate, targets[:, :].cuda())
                        
                        # Iterative Data
                        # _, perturb = pgd_attack.perturb_iter_data(image_to_animate, all_images, image_to_animate, targets[68, :].unsqueeze(0).cuda())

                    # Normal Attack
                    x_adv, perturb = pgd_attack.perturb(image_to_animate, resulting_image_noattack, targets_au)

                    # Use this line if transferring attacks
                    x_adv = image_to_animate + perturb

                    # No Attack
                    # x_adv = image_to_animate

                    with torch.no_grad():
                        resulting_images_att, resulting_images_reg = self.G(
                            x_adv, targets_au)
                        resulting_image = self.imFromAttReg(
                            resulting_images_att, resulting_images_reg, x_adv).cuda()

                    save_image((resulting_image+1)/2, os.path.join(self.animation_results_dir,
                                                                   image_path.split('/')[-1].split('.')[0]
                                                                   + '_' + reference_expression_images[target_idx]))
                    if target_idx == 0:
                        save_image((x_adv+1)/2, os.path.join(self.animation_results_dir,
                                                                   image_path.split('/')[-1].split('.')[0]
                                                                   + '_ref.jpg'))

                    # Compare to ground-truth output
                    l1_error += F.l1_loss(resulting_image, resulting_image_noattack)
                    l2_error += F.mse_loss(resulting_image, resulting_image_noattack)
                    l0_error += (resulting_image - resulting_image_noattack).norm(0)
                    min_dist += (resulting_image - resulting_image_noattack).norm(float('-inf'))

                    # Compare to input image
                    # l1_error += F.l1_loss(resulting_image, x_adv)
                    # l2_error += F.mse_loss(resulting_image, x_adv)
                    # l0_error += (resulting_image - x_adv).norm(0)
                    # min_dist += (resulting_image - x_adv).norm(float('-inf'))

                    if F.mse_loss(resulting_image, resulting_image_noattack) > 0.05:
                        n_dist += 1
                    n_samples += 1

                    # Debug
                    # x_adv, targets_au, resulting_image, resulting_images_att, resulting_images_reg = None, None, None, None, None
                
                image_to_animate = None

            # Print metrics
            print('{} images. L1 error: {}. L2 error: {}. prop_dist: {}. L0 error: {}. L_-inf error: {}.'.format(n_samples, 
            l1_error / n_samples, l2_error / n_samples, float(n_dist) / float(n_samples), l0_error / n_samples, min_dist / n_samples))
