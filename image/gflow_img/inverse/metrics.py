from utils.misc import deterministic, set_cuda_visible_device
from utils.utils_score import FidScore, LpipsScore,PSNRScore,SSIMScore
import os
import tyro
from config.inverse_problem import InverseProblemConfig


def get_image_paths(path):
    images = []
    for root,dirs,files in os.walk(path):
        for file in files:
            images.append(os.path.join(root, file))
    return images

def compute_metrics(cfg: InverseProblemConfig, path_generated, path_real):
    generated_image_paths = get_image_paths(path_generated)
    real_image_paths = get_image_paths(path_real)
    generated_image_paths.sort()
    real_image_paths.sort()
    assert len(generated_image_paths) == len(real_image_paths)

    fid = FidScore('cuda',batchsize=128)
    fid_score = fid.compute(real_image_paths,generated_image_paths)
    print("FID Score:", fid_score.item())

    lpip = LpipsScore('cuda',batch_size=128)
    lpip_score = lpip.compute(real_image_paths,generated_image_paths)
    print("LPIPS Score:", lpip_score.item())

    psnr = PSNRScore('cuda',batch_size=256)
    psnr_score = psnr.compute(real_image_paths,generated_image_paths)
    print("PSNR Score:", psnr_score.item())

    ssim = SSIMScore('cuda',batch_size=256)
    ssim_score = ssim.compute(real_image_paths,generated_image_paths)
    print("SSIM Score:", ssim_score.item())

    return fid_score, lpip_score, psnr_score, ssim_score