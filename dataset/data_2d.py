import numpy as np
import torch
from torch.utils.data import Dataset
import pdb
import sys, os

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

from IPython import embed

class Smoke(Dataset):
    def __init__(
        self,
        dataset_path,
        time_steps=64,
        horizon=10,
        time_interval=1,
        all_size=128,
        size=64,
        is_train=True,
    ):
        super().__init__()
        self.root = dataset_path
        self.time_steps = time_steps # total time steps of each trajectory after down sampling
        self.horizon = horizon # horizon of diffusion model
        self.time_interval = time_interval
        self.time_steps_effective = (self.time_steps - self.horizon + 1) // self.time_interval
        self.all_size = all_size
        self.size = size
        self.space_interval = int(all_size/size)
        self.is_train = is_train
        self.dirname = "train" if self.is_train else "test"
        if self.is_train:
            self.n_simu = 40000
        else:
            self.n_simu = 50
        # self.RESCALER = torch.tensor([3, 20, 20, 17, 19, 1]).reshape(1, 6, 1, 1) 
        self.RESCALER = torch.tensor([1, 45, 50, 45, 50, 1]).reshape(1, 6, 1, 1)  # rescale the data to [-1, 1] with relaxation, on 64 time steps dataset

    def __len__(self):
        # return self.n_simu
        if self.is_train:
            return self.n_simu * self.time_steps_effective
        else:
            return self.n_simu

    def __getitem__(self, idx):
        if self.is_train:
            sim_id, time_id = divmod(idx, self.time_steps_effective)
        else:
            sim_id, time_id = idx, 0 # for test, pass each trajectory as a whole and only once

        if self.is_train:
            d = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1) # 2, 65, 64, 64
            c = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            s = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                             dtype=torch.float) # 65, 8
            s = s[:, 1]/s.sum(-1) # shape: [65]; 1 is index of of the target bucket
            s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], self.size, self.size) # 1, 65, 64, 64
            state = torch.cat((d, v, c, s), dim=0)[:, time_id: time_id + self.horizon] # 6, horizon, 64, 64
        
            data = (
                state.permute(1, 0, 2, 3) / self.RESCALER, # horizon, 6, 64, 64
                list(state.shape[-3:]),
                list(state.shape[-3:]),
                sim_id,
            )
        else:
            d = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            c = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            s = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                             dtype=torch.float)
            
            s = s[:, 1]/s.sum(-1)
            s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], self.size, self.size) 
            state = torch.cat((d, v, c, s), dim=0) # 6, 65, 64, 64
            data = (
                state.permute(1, 0, 2, 3), # 65, 6, 64, 64, not rescaled
                list(state.shape[-3:]),
                list(state.shape[-3:]),
                sim_id,
            )

        return data

if __name__ == "__main__":
    dataset = Smoke(
        dataset_path="/data/",
        is_train=True,
    )
    print("len(dataset): ", len(dataset))