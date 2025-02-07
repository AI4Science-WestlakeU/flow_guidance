import tyro
import os
from tqdm import tqdm
import torch
from gflow_img.dataset.dataset import get_dataloader
from gflow_img.config.inverse_problem import InverseProblemConfig
from gflow_img.utils.misc import deterministic, set_cuda_visible_device
from gflow_img.inverse.inverse_problems import InverseProblem
from run.compute_metrics import compute_save_metrics

# utils
def run(ip: InverseProblem, cfg: InverseProblemConfig):
    os.makedirs(os.path.join(cfg.make_path(), 'x_gen'), exist_ok=True)
    os.makedirs(os.path.join(cfg.make_path(), 'x_gt'), exist_ok=True)
    os.makedirs(os.path.join(cfg.make_path(), 'y'), exist_ok=True)
    
    # prepare dataset
    if cfg.use_dataset:
        train_dl, val_dl, test_dl = get_dataloader(
            ip.data_name, cfg.batch_size, data_cache_path=cfg.data_cache_dir
        )
    if cfg.guide_method == 'MC':
        print("loading mc support dataset")
        mc_dl, _, _ = get_dataloader(
            ip.data_name, cfg.mc_batch_size, data_cache_path=cfg.data_cache_dir
        )
        data_x1 = next(iter(mc_dl))
    
    # run inference
    ip.reset_counter()
    ip.remove_cached_x1_for_mc()
    for data in tqdm(val_dl):
        if cfg.guide_method in ['PiGDM', 'PiGDM+']:
            with torch.no_grad():
                ip.solve_ip_pgdm(data)
        elif cfg.guide_method in ['nabla_xt_J_xt', 'nabla_x1_J_x1', 'nabla_xt_J_x1']:
            with torch.no_grad():
                ip.solve_ip_grad(data)
        elif cfg.guide_method == 'MC':
            with torch.no_grad():
                ip.solve_ip_mc(data, data_x1)
            ip.remove_cached_j_for_mc() # need to remove cached Jx1 for this batch
        else:
            raise ValueError(f"Unknown guide function: {cfg.guide_method}")
        
    print('Generation Finished. Computing metrics...')
    compute_save_metrics(cfg)


if __name__ == '__main__':
    cfg = tyro.cli(InverseProblemConfig)
    deterministic(cfg.seed)
    set_cuda_visible_device(cfg)

    run(InverseProblem(cfg), cfg)
