import os
import yaml


def deterministic(seed):
    import torch
    import numpy
    import random
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

def save_config(cfg, exp_dir, config_name='config.yaml'):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    else:
        print(f"[ Initialize Logger ] Will save model to {exp_dir}, old cps will be overwritten!")
    with open(os.path.join(exp_dir, config_name), 'w') as f:
        yaml.dump(vars(cfg), f)
