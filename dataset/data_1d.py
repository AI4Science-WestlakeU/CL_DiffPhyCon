import scipy.io
import numpy as np
import h5py
import pdb
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from IPython import embed

def get_burgers_preprocess(
    rescaler=None, 
    stack_u_and_f=False, 
    pad_for_2d_conv=False, 
    partially_observed_fill_zero_unobserved=None, 
):
    if rescaler is None:
        raise NotImplementedError('Should specify rescaler. If no rescaler is not used, specify 1.')
    
    def preprocess(db):
        '''We are only returning f and u for now, in the shape of 
        (u0, u1, ..., f0, f1, ...)
        '''
        
        u = db['u']
        f = db['f']
        f = f[:,:15]

        fill_zero_unobserved = partially_observed_fill_zero_unobserved
        if fill_zero_unobserved is not None:
            if fill_zero_unobserved == 'front_rear_quarter':
                u = u.squeeze()
                nx = u.shape[-1]
                u[..., nx // 4: (nx * 3) // 4] = 0
            else:
                raise ValueError('Unknown partially observed mode')
        
        if stack_u_and_f:
            assert pad_for_2d_conv
            nt = f.size(-2)
            f = nn.functional.pad(f, (0, 0, 0, 16 - nt), 'constant', 0)
            u = nn.functional.pad(u, (0, 0, 0, 15 - nt), 'constant', 0)
            u_target = u  
            data = torch.stack((u, f, u_target), dim=1) 
        else:
            assert not pad_for_2d_conv
            data = torch.cat((u, f), dim=1)
    
        data = data / rescaler
        return data

    return preprocess



class DiffusionDataset(Dataset):
    def __init__(
        self, 
        fname, 
        preprocess=get_burgers_preprocess('all'),  
        load_all=True
    ):
        '''
        Arguments:

        '''
        self.load_all = load_all
        if load_all:
            self.db = torch.load(fname)
            self.x = preprocess(self.db)
        else:
            raise NotImplementedError

    def __len__(self):
        if self.load_all:
            return self.x.size(0)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.load_all:
            return self.x[idx]
        else:
            raise NotImplementedError

    def get(self, idx):
        return self.__getitem__(idx)
    
    def len(self):
        return self.__len__()