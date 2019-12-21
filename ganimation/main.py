import os

from config import get_config
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True  # Improves runtime if the input size is constant

    config.outputs_dir = os.path.join('experiments', config.outputs_dir)

    config.log_dir = os.path.join(config.outputs_dir, config.log_dir)
    config.model_save_dir = os.path.join(
        config.outputs_dir, config.model_save_dir)
    config.sample_dir = os.path.join(config.outputs_dir, config.sample_dir)
    config.result_dir = os.path.join(config.outputs_dir, config.result_dir)

    data_loader = get_loader(config.image_dir, config.attr_path, config.c_dim,
                             config.image_size, config.batch_size, config.mode,
                             config.num_workers)

    config_dict = vars(config)
    solver = Solver(data_loader, config_dict)

    if config.mode == 'train':
        initialize_train_directories(config)
        solver.train()
    elif config.mode == 'animation':
        initialize_animation_directories(config)
        solver.animation()


def initialize_train_directories(config):
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)


def initialize_animation_directories(config):
    if not os.path.exists(config.animation_results_dir):
        os.makedirs(config.animation_results_dir)


if __name__ == '__main__':

    config = get_config()
    print(config)

    main(config)
