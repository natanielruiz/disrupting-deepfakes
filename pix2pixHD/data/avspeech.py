import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from data import landmarks
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import glob
import random

class AVSpeech(Dataset):
    def __init__(self, transform):
        self.frame_folder = 'datasets/avspeech/frames'
        self.meta_folder = 'datasets/avspeech/meta'
        self.user_folders = glob.glob(os.path.join(self.frame_folder, '*'))
        self.users = [x.split('/')[-1] for x in self.user_folders]

        self.transform = transform

        self.length = len(self.users)

    def __getitem__(self, index):
        # Get list of frames for user
        user = self.users[index]
        meta = dict(np.load(os.path.join(self.meta_folder, '{}.npz'.format(user))))

        frame_list = glob.glob(os.path.join(self.frame_folder, '{}/*.png'.format(user)))
        frame_list = [int(x.split('/')[-1].split('.')[0]) for x in frame_list]

        ref_frame = random.choice(frame_list)
        tgt_frame = random.choice(frame_list)

        ref_img = Image.open(os.path.join(self.frame_folder, '{}/{}.png'.format(user, ref_frame)))
        tgt_img = Image.open(os.path.join(self.frame_folder, '{}/{}.png'.format(user, tgt_frame)))

        # Make reference and target landmarks
        ref_lnd = landmarks.plot_landmarks(landmarks.get_relative_landmarks(meta, ref_frame))
        tgt_lnd = landmarks.plot_landmarks(landmarks.get_relative_landmarks(meta, tgt_frame))

        ref_img = self.transform(ref_img)
        tgt_img = self.transform(tgt_img)
        ref_lnd = self.transform(ref_lnd)
        tgt_lnd = self.transform(tgt_lnd)

        input_dict = {'ref_img': ref_img, 'tgt_img': tgt_img, 'ref_lnd': ref_lnd, 
                      'tgt_lnd': tgt_lnd, 'user': user}

        return input_dict

    def __len__(self):
        return self.length