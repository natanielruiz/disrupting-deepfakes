from .base_model import BaseModel
from . import networks
from util import attacks
import numpy as np
import torch
import torch.nn.functional as F

class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake', 'fake_noattack']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real)  # G(real)

    def forward_noattack(self):
        """Run forward pass."""
        self.fake_noattack = self.netG(self.real)  # G(real)

    def attack(self, target):
        image = self.real
        
        # Attack
        pgd_attack = attacks.LinfPGDAttack(model=self.netG)
        input_adv, perturb = pgd_attack.perturb(image, target)      
        
        return input_adv, perturb

    def forward_attack(self, perturb):
        self.real = torch.clamp(self.real + perturb, min=-1, max=1)   
        self.fake = self.netG(self.real)  # G(real)

    def compute_errors(self):
        generated = self.fake
        generated_noattack = self.fake_noattack
        l1 = F.l1_loss(generated, generated_noattack)
        l2 = F.mse_loss(generated, generated_noattack)
        l0 = (generated - generated_noattack).norm(0)
        d = (generated - generated_noattack).norm(float('-inf'))

        if F.mse_loss(generated, generated_noattack) > 0.05:
            n_dist = 1
        else:
            n_dist = 0
        
        return l1, l2, l0, d, n_dist

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
