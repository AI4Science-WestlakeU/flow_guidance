import torch
import torchdiffeq
from gflow_img.backbone.unet_configs import instantiate_model

class Sample():
    def __init__(self, model_name, weight_file, device):
        self.device = device
        self.model_name = model_name
        self.model = instantiate_model(model_name, False)
        self.model.load_state_dict(torch.load(weight_file))
        self.model.to(self.device)
        self.model.eval()

    def sample(self,batch_size=8,size=64,save_file='test.pt',steps=10,method='dopri5'):

        with torch.no_grad():
            traj = torchdiffeq.odeint(
                        lambda t, x: self.model.forward(t, x),
                        torch.randn(batch_size, 3, size, size, device=self.device),
                        torch.linspace(0, 1, steps, device=self.device),
                        atol=1e-4,
                        rtol=1e-4,
                        method="dopri5",
                    )
            torch.save(traj, save_file)
            return traj.detach().cpu().numpy()

if __name__ == '__main__':

    # sample = Sample('unet128', 'results/unet128_imagenet128/model_2.pt', 'cuda:0')
    # sample.sample(8, 128, './unet128_imagenet_traj.pt',steps=100)

    sample = Sample('punet256', 'results/cfm_punet256_celeba256/model_499.pt', 'cuda:0')
    sample.sample(8, 256, './unet256_celeba_traj.pt',steps=1000,method='euler')
