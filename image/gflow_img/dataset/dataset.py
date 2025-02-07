"""Return training and evaluation/test datasets from config files."""
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import json
from torchvision import transforms
import pdb
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x

class CelebAHQDataset(Dataset):
    def __init__(self, image_files,
                root_dir = 'data_cache/celeba_hq_256',
                transform=None,
                resize = False):
        """
        Args:
            image_files (list): List of image file names.
            root_dir (str): Path to the directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_files = image_files
        self.root_dir = root_dir
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        if self.resize == True:
            image = image.resize((128, 128))
        if self.transform:
            image = self.transform(image)
        return image
    

def train_test_vali_split(data_dir,split_file):
    image_files = sorted(os.listdir(data_dir))
    train_files, temp_files = train_test_split(image_files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")
    splits = {
        "train": train_files,
        "validation": val_files,
        "test": test_files
    }
    with open(split_file, "w") as f:
        json.dump(splits, f)
    return 


def load_existing_split(data_dir, split_file, transform=None,resize=False):
    """
    Loads existing splits from a JSON file and creates datasets.
    
    Args:
        data_dir (str): Path to the directory containing the images.
        split_file (str): Path to the JSON file containing the splits.
        transform (callable): Transformations to apply to the images.
    
    Returns:
        dict: A dictionary with "train", "validation", and "test" datasets.
    """

    with open(split_file, "r") as f:
        split_info = json.load(f)
    

    datasets = {
        "train": CelebAHQDataset(split_info["train"], data_dir, transform=transform,resize=resize),
        "validation": CelebAHQDataset(split_info["validation"], data_dir, transform=transform,resize=resize),
        "test": CelebAHQDataset(split_info["test"], data_dir, transform=transform,resize=resize),
    }
    
    return datasets


class ImageNet(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        temp = self.data[idx]
        label = temp['label']
        img = temp['image']
        if self.transform:
            img = self.transform(img)
        return img
    
def get_dataloader(data_name, batch_size, transform=None, data_cache_path='.'):
    transform = transforms.Compose([
            transforms.ToTensor(), 
        ])
    if data_name == 'celeba128':
        resize = True
    else:
        resize = False
    if data_name.startswith('celeba'):
        data_dir = os.path.join(data_cache_path, 'data_cache/celeba_hq_256')
        splits_file = os.path.join(data_cache_path, "data_cache/celeba_hq_splits.json")
        
        if os.path.exists(path=splits_file):
            datasets = load_existing_split(data_dir, splits_file,transform,resize=resize)
        else:
            train_test_vali_split(data_dir, splits_file)
            datasets = load_existing_split(data_dir, splits_file,transform,resize=resize)
        train_dl = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(datasets['validation'], batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)
    else:
        raise NotImplementedError(f"Data loader for {data_name} not implemented.")
    return train_dl,val_dl,test_dl


if __name__ == "__main__":
    get_dataloader('celeba',8,resize=True)