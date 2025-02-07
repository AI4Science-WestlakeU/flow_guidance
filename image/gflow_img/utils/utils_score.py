import torch.nn as nn
from functools import partial

def get_criterion(name):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif name == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f'Criterion {name} not supported')

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

class FidDataset(Dataset):
    def __init__(self, image_files,fid=False):

        self.image_files = image_files
        if fid:
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("RGB")

        image = self.transform(image)
        return image

class FidScore():
    def __init__(self,device,batchsize=256):
        self.fid = FrechetInceptionDistance(
            feature=2048,        
            reset_real_features=False,
            normalize=True,                
        ).to(device)
        self.batchsize = batchsize
        self.device = device                  

    def update(self, real_loader, fake_loader):
        for real_batch in tqdm(real_loader):
            real_batch = real_batch.to(self.device)
            self.fid.update(real_batch, real=True)

        for fake_batch in tqdm(fake_loader):
            fake_batch = fake_batch.to(self.device)
            self.fid.update(fake_batch, real=False)

    def compute(self,real_image_path,fake_image_path):
        real_dataset = FidDataset(image_files=real_image_path,fid=True)
        fake_dataset = FidDataset(image_files=fake_image_path,fid=True)

        real_loader = DataLoader(real_dataset, batch_size=self.batchsize)
        fake_loader = DataLoader(fake_dataset, batch_size=self.batchsize)
        self.update(real_loader,fake_loader)
        return self.fid.compute()


class PairDataset(Dataset):
    def __init__(self, real_image_path, gen_image_path):
        self.real_image_path = real_image_path
        self.gen_image_path = gen_image_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.real_image_path)

    def __getitem__(self, idx):
        real_image_name = self.real_image_path[idx]
        real_image = Image.open(real_image_name).convert("RGB")
        real_image = self.transform(real_image)

        gen_image_name = self.gen_image_path[idx]
        gen_image = Image.open(gen_image_name).convert("RGB")
        gen_image = self.transform(gen_image)
        return real_image, gen_image

class LpipsScore():
    def __init__(self,device,batch_size=16):
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
            reduction='mean'
        ).to(device)
        self.batch_size = batch_size
        self.device = device

    def compute(self,real_image_path,fake_image_path):
        dataset = PairDataset(real_image_path,fake_image_path)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        for real_imgs, gen_imgs in tqdm(data_loader):
            real_imgs = real_imgs.to(self.device)
            gen_imgs  = gen_imgs.to(self.device)

            self.lpips.update(gen_imgs,real_imgs)
        return self.lpips.compute()

class PSNRScore():
    def __init__(self,device,data_range=1.0,batch_size=16):
        self.psnr = PeakSignalNoiseRatio(data_range=data_range).to(device) 
        self.device = device
        self.batch_size = batch_size
    def compute(self,real_image_path,fake_image_path):
        dataset = PairDataset(real_image_path,fake_image_path)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        for real_imgs, gen_imgs in tqdm(data_loader):
            real_imgs = real_imgs.to(self.device)
            gen_imgs  = gen_imgs.to(self.device)
            self.psnr.update(gen_imgs,real_imgs)
        return self.psnr.compute()

class SSIMScore():
    def __init__(self,device,data_range=1.0,batch_size=16):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(device) 
        self.device = device
        self.batch_size = batch_size
    def compute(self,real_image_path,fake_image_path):
        dataset = PairDataset(real_image_path,fake_image_path)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        for real_imgs, gen_imgs in tqdm(data_loader):
            real_imgs = real_imgs.to(self.device)
            gen_imgs  = gen_imgs.to(self.device)
            self.ssim.update(gen_imgs,real_imgs)
        return self.ssim.compute()


if __name__ == '__main__':
    real_image_path = ['data_cache/celeba_hq_256/00003.jpg',
                        'data_cache/celeba_hq_256/00003.jpg',
                        'data_cache/celeba_hq_256/00003.jpg',]
    fake_image_path = ['data_cache/celeba_hq_256/00003.jpg',
                        'data_cache/celeba_hq_256/00003.jpg',
                        'data_cache/celeba_hq_256/00000.jpg',]
    
    # calculate FID example
    fid_score = FidScore('cuda')
    score = fid_score.compute(real_image_path,fake_image_path)
    print("FID Score:", score.item())

    # calculate LPIPS example
    lpip = LpipsScore('cuda')
    score = lpip.compute(real_image_path,fake_image_path)
    print("LPIPS Score:", score.item())

    # psnr score example
    psnr = PSNRScore('cuda')
    score = psnr.compute(real_image_path,fake_image_path)
    print("PSNR Score:", score.item())

    # ssim score example
    ssim = SSIMScore('cuda')
    score = ssim.compute(real_image_path,fake_image_path)
    print("SSIM Score:", score.item())