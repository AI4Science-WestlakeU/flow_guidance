import torch
from torch import nn
from tqdm import tqdm

from torchdyn.core import NeuralODE
import torchdiffeq

class Trainer():
    def __init__(self,FM, model, optimizer,accelerator, lr_scheduler, criterion,train_dl,val_dl, device, results_folder,accumulation_steps=1):
        self.FM = FM
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.accelerator = accelerator
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.lr_scheduler = lr_scheduler
        self.results_folder = results_folder 
        self.accumulation_steps = accumulation_steps
        # self.fid_score = fid_score(self.val_dl, self.device)
        # self.node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    
    def train_step(self, data):
        self.optimizer.zero_grad()
        
        x1 = data.to(self.device)
        
        x0 = torch.randn_like(x1)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
        vt = self.model(t, xt)

        loss = self.criterion(vt, ut)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train_loop(self, num_epochs,strat_epoch=0):
        self.model.train()

        for epoch in range(strat_epoch,num_epochs):
            losses = []
            for i, data in enumerate(tqdm(self.train_dl)):
                loss = self.train_step(data)
                losses.append(loss)
                
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            print(f'Epoch {epoch}, loss {torch.tensor(losses).mean()}')
            if self.accelerator.is_main_process:
                trajs = self.sample(size=data.shape[-1])
                torch.save(trajs, f'{self.results_folder}/trajs_{epoch}.pt')
                torch.save(self.accelerator.get_state_dict(self.model), f'{self.results_folder}/model_{epoch}.pt')

    def sample(self,batch_size=4,size=64):
        USE_TORCH_DIFFEQ = True
        with torch.no_grad():
            if USE_TORCH_DIFFEQ:
                traj = torchdiffeq.odeint(
                    lambda t, x: self.model.forward(t, x),
                    torch.randn(batch_size, 3, size, size, device=self.device),
                    torch.linspace(0, 1, 1000, device=self.device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5",
                )
            else:
                raise NotImplementedError("Not implemented")
            # else:
            #     traj = self.node.trajectory(
            #         torch.randn(100, 1, 28, 28, device=self.device),
            #         t_span=torch.linspace(0, 1, 2, device=self.device),
            #     )
        return traj.detach().cpu()