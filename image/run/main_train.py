import json
import os
import sys
from dataclasses import dataclass

from gflow_img.utils.utils_score import get_criterion
from gflow_img.utils.misc import deterministic, set_cuda_visible_device
import numpy as np
import torch
import tyro
from accelerate import Accelerator

from gflow_img.backbone.unet_configs import *
from gflow_img.cfm.conditional_flow_matching import *
from gflow_img.dataset.dataset import get_dataloader
from gflow_img.utils.trainer import Trainer
from gflow_img.cfm.conditional_flow_matching import *
from gflow_img.backbone.unet_configs import *
from gflow_img.config.training import TrainingConfig

def train(cfg: TrainingConfig):
    # model
    model_name = cfg.model_data.split('_')[0]
    model = instantiate_model(model_name, cfg.use_ema)
    if cfg.load_pretrained_file != None:
        model.load_state_dict(torch.load(cfg.load_pretrained_file))
        print('load pretrained model from',cfg.load_pretrained_file)
    # optimizer
    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError('optimizer not implemented')

    # learning rate scheduler
    if cfg.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.num_epochs)
    elif cfg.lr_scheduler == 'None':
        lr_scheduler = None
    else:
        raise NotImplementedError('lr scheduler not implemented')

    # criterion
    criterion = get_criterion(cfg.criterion)

    # dataloader
    data_name = cfg.model_data.split('_')[1]
    train_dl,val_dl,test_dl = get_dataloader(data_name, cfg.batch_size, data_cache_path=cfg.data_cache_dir)

    # accelerator
    accelerator = Accelerator(
        split_batches = True, # if True, then actual batch size equals args.batch_size
    )
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    # flow maching
    sigma = 0.01
    if cfg.flow_type == 'ot':
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma)
    elif cfg.flow_type == 'cfm':
        FM = ConditionalFlowMatcher(sigma)
    else:
        raise NotImplementedError('flow type not implemented')

    # for train
    trainer = Trainer(
                    FM=FM,
                    model=model,
                    optimizer=optimizer,
                    accelerator = accelerator,
                    lr_scheduler=lr_scheduler,
                    criterion=criterion,
                    train_dl=train_dl,
                    val_dl=val_dl,
                    device=accelerator.device,
                    results_folder=os.path.join('results', cfg.flow_type + '_' + cfg.model_data),
                    accumulation_steps=cfg.accumulation_steps)
    if cfg.load_pretrained_file != None:
        start_epoch = cfg.load_pretrained_file.split('_')[-1].split('.')[0]
        start_epoch = int(start_epoch) + 1
    else:
        start_epoch = 0

    trainer.train_loop(cfg.num_epochs, strat_epoch=start_epoch)

if __name__ == '__main__':
    cfg = tyro.cli(TrainingConfig)
    # Set the environment variable to use GPU 0
    set_cuda_visible_device(cfg)
    os.makedirs(os.path.join('results', cfg.flow_type + '_' + cfg.model_data), exist_ok=True)
    # set seed
    deterministic(cfg.seed)
    train(cfg)