import os
import yaml
import random
import numpy
import torch


def save_config(cfg, exp_dir):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    else:
        print(f"save model to {exp_dir}, old configs, checkpoints will be overwritten; out evaluation results will be appended")
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(cfg), f)

def deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    # torch deterministic algorithm
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_cuda_visible_device(cfg, outmost=True, debug=False):
    """Set environment variable CUDA_VISIBLE_DEVICES to outmost cfg.device, 
    then set cfg.device and cfg.*.device to 'cuda:0' recursively"""
    # cuda visible device
    if outmost:
        import os
        cuda_num = cfg.device.split(':')[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num
    
    # set device to cuda:0
    if debug:
        print(f'in: {type(cfg)}')
    
    for key, value in cfg.__dict__.items():
        if key == 'device':
            cfg.device = 'cuda:0'

        if hasattr(value, '__dict__'):
            set_cuda_visible_device(value, outmost=False)
