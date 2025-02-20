# Adapted from https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers.git
# Adapted from 'Solving PDE-constrained Control Problems using Operator Learning'

import argparse
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import h5py
import random
from copy import copy
from datetime import datetime
import tqdm
from IPython import embed
import yaml
sys.path.append('.')
from utils_1d.result_io import merge_save_dict

VISC = 0.01 # default viscousity

class burgers():
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 L: float=None,
                 flux_splitting: str=None,
                 device: torch.cuda.device = "cpu") -> None:
        """
        Args:
            tmin (float): starting time
            tmax (float): end time
            grid_size (list): grid points [nt, nx]
            L (float): periodicity
            device (torch.cuda.device): device (cpu/gpu)
        Returns:
            None
        """
        # Data params for grid and initial conditions
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 1 if tmax is None else tmax
        # Length of the spatial domain / periodicity
        self.L = 1 if L is None else L
        self.grid_size = (11, 128) if grid_size is None else grid_size
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1]+1)
        self.device = device
        self.force = None

    def __repr__(self):
        return f'burgers'


def check_files(save_path) -> None:
    """
    Check if data files exist. Will NOT replace them.
    """
    train_name = os.path.join(save_path, 'train')
    test_name = os.path.join(save_path, 'test')
    print(train_name)
    if os.path.isfile(train_name) or os.path.isfile(test_name):
        raise FileExistsError('File already exists. Remove both train and test sets before generating new ones.')

def check_directory(file_path) -> None:
    """
    Check if data and log directories exist, and create otherwise
    Args:
    Returns:
        None
    """
    if os.path.exists(file_path):
        print(f'Data directory exists and will be written to.')
    else:
        os.mkdir(file_path)
        print(f'Data directory created.')
    

def Diff_mat_1D(Nx, device='cpu'):
    # I tried to change the implementation to torch dense matrix here, but kept the original implem.
    # First derivative
    D_1d = sp.diags([-1, 1], [-1, 1], shape = (Nx,Nx)) # A division by (2*dx) is required later.
    D_1d = sp.lil_matrix(D_1d)
    D_1d[0,[0,1,2]] = [-3, 4, -1]               # this is 2nd order forward difference (2*dx division is required)
    D_1d[Nx-1,[Nx-3, Nx-2, Nx-1]] = [1, -4, 3]  # this is 2nd order backward difference (2*dx division is required)

    # Second derivative
    D2_1d = sp.diags([1, -2, 1], [-1,0,1], shape = (Nx, Nx)) # division by dx^2 required
    D2_1d = sp.lil_matrix(D2_1d)                  
    D2_1d[0,[0,1,2,3]] = [2, -5, 4, -1]                    # this is 2nd order forward difference. division by dx^2 required. 
    D2_1d[Nx-1,[Nx-4, Nx-3, Nx-2, Nx-1]] = [-1, 4, -5, 2]  # this is 2nd order backward difference. division by dx^2 required.
    

    return D_1d, D2_1d


def burgers_numeric_solve_free(u0, f, visc, T, dt=1e-4, num_t=10, mode=None):
    '''
    Simulates trajectories based on u0 and f. Trajectory i is based on u0[i, :]
    and f[i, :, :]

    Args:
        u0: (N,s), N is the number of samples (every sample has different u0 and f)
        f: (N,Nt,s)
        T: physical simulation time
        dt: physical simulation stepsize
        num_t: 
            number of sampling of times
            num of controllable forces (forces f_[i: i + T / dt] are the same)
    Returns:
        simulated u: (N_{u0 and f}, num_t, n_spatial_grids)
    '''
    if mode!='const':
        assert f.size()[1]==num_t, 'check number of time interval'
    else:
        raise ValueError


    #Grid size
    s = u0.size(-1)
    Nt = f.size(1)

    Nu0 = u0.size(0)
    Nf = f.size(0)
    assert Nu0 == Nf
    N = Nf
     
    xmin = 0.0; xmax = 1.0
    delta_x = (xmax-xmin)/(s+1)

    #Number of steps to final time
    steps = math.ceil(T/dt)
    
    u = u0.reshape(N, s)
    
    u = F.pad(u, (1,1))
    
    f = f.reshape(N, Nt, s)
    
    f = F.pad(f, (1,1))

    #Record solution every this number of steps
    record_time = math.floor(steps / Nt)
        
    D_1d, D2_1d = Diff_mat_1D(s + 2, device=u0.device)
    D_1d.rows[0] = D_1d.rows[0][:2]
    D_1d.rows[-1] = D_1d.rows[-1][-2:]
    D_1d.data[0] = D_1d.data[0][:2]
    D_1d.data[-1] = D_1d.data[-1][-2:]
    
    D2_1d.rows[0] = D2_1d.rows[0][:3]
    D2_1d.rows[-1] = D2_1d.rows[-1][-3:]
    D2_1d.data[0] = D2_1d.data[0][:3]
    D2_1d.data[-1] = D2_1d.data[-1][-3:]
    
    t_sys_ind = list(D_1d.rows)
    t_sys = torch.FloatTensor(np.stack(D_1d.data)/(2*delta_x)).to(u0.device)
    d_sys_ind = list(D2_1d.rows)
    d_sys = torch.FloatTensor(visc*np.stack(D2_1d.data)/delta_x**2).to(u0.device)
    
    #Saving solution and time
    sol = torch.zeros(N, s, Nt, device=u0.device)
    
    #Record counter
    c = 0
    #Physical time
    t = 0.0
    f_idx = -1
    print(steps, record_time)
    for j in tqdm.trange(steps):
        u = u[...,1:-1]
        u = F.pad(u, (1,1))
        
        u_s = u**2
        transport = torch.einsum('nsi,si->ns', u_s[...,t_sys_ind], t_sys)
        diffusion = torch.einsum('nsi,si->ns', u[...,d_sys_ind], d_sys)
        
        
        if j % record_time == 0:
            f_idx += 1
        u = u + dt * (-(1 / 2) * transport + diffusion + f[:, f_idx, :])
        
        #Update real time (used only for recording)
        t += dt

        if (j+1) % record_time == 0:

            #Record solution and time
            sol[...,c] = u[...,1:-1]
            c += 1

    sol = sol.permute(0, 2, 1) # (N, Nt, Nx)
    trajectory = torch.cat((u0.reshape(N, 1, s), sol), dim=1) # (N, Nt + 1, Nx)
    
    return trajectory


def make_data_varying_f(Nu0, Nf, s, t, amp_compensate=2, partial_control=None, alpha=1.):
    '''
    Arguments:
        amp_compensate: 
        Gaussian in the time domain decreases average amp, so we need to compensate it
        t: number of time stamps of f (In the Burgers dataset this stands for the "ladders"
        of the control sequence.
        )

    If s=num_interior=128 // num_pts=130=0,dx,...,129*dx // dx=1/129
    
    Returns:
        u0 (N, Nx), numpy array. 
        f (N, Nt, Nx), torch tensor. float 32
    '''
    
    xmin = 0.0; xmax = 1.0
    delta_x = (xmax-xmin)/(s+1)
    x = torch.linspace(xmin+delta_x,xmax-delta_x,s)

    tmin = 0.0; tmax = 1.0
    delta_t = (tmax-tmin)/(t+1)
    ts = torch.linspace(tmin+delta_t,tmax-delta_t,t)
    
    #make u0
    loc1 = np.random.uniform(0.2, 0.4, (Nu0,1))
    amp1 = np.random.uniform(0, 2, (Nu0,1))
    sig1 = np.random.uniform(0.05, 0.15, (Nu0,1))
    gauss1=amp1*np.exp(-0.5*(np.array(x.view(1,-1).repeat(Nu0,1))-loc1)**2/sig1**2)

    loc2 = np.random.uniform(0.6, 0.8, (Nu0,1))
    amp2 = np.random.uniform(-2, 0, (Nu0,1))
    sig2 = np.random.uniform(0.05, 0.15, (Nu0,1))
    gauss2=amp2*np.exp(-0.5*(np.array(x.view(1,-1).repeat(Nu0,1))-loc2)**2/sig2**2)

    u0=gauss1+gauss2
    
    # make f
    # partial control
    if partial_control is None:
        f_space_mask = np.ones_like(x.view(1, 1, -1).repeat(Nf, t, 1))
    elif partial_control == 'front_rear_quarter':
        f_space_mask = np.zeros_like(x.view(1, 1, -1).repeat(Nf, t, 1))
        controllable_idx = np.hstack((
            np.arange(0, s // 4), 
            np.arange(3 * s // 4, s)
        ))
        f_space_mask[:, :, controllable_idx] = 1.
        # 
        amp_compensate *= 2
        print('Generating in partial control mode:', partial_control)
    else:
        raise ValueError('invalid partial control mode')

    def rand_f(is_rand_amp=True):
        if is_rand_amp:
            amp = np.random.randint(2, size=(Nf, 1, 1)) * np.random.uniform(-1.5, 1.5, (Nf, 1, 1))
        else:
            amp = np.random.uniform(-1.5, 1.5, (Nf, 1, 1))
        amp = torch.tensor(amp).repeat(1, t, s)

        loc = np.random.uniform(0, 1, (Nf, 1, 1))
        sig = np.random.uniform(0.1, 0.4, (Nf, 1, 1))*0.5
        exp_space = np.exp(-0.5*(np.array(x.view(1, 1, -1).repeat(Nf, t, 1))-loc)**2/sig**2)
        exp_space = exp_space * f_space_mask

        loc = np.random.uniform(0, 1, (Nf, 1, 1))
        sig = np.random.uniform(0.1, 0.4, (Nf, 1, 1)) * 0.5
        exp_time = amp_compensate * \
            np.exp(-0.5*(np.array(ts.view(1, -1, 1).repeat(Nf, 1, s))-loc)**2/sig**2)
        return amp * exp_space * exp_time
    
    sum_num_f=7
    f=rand_f(is_rand_amp=False) # To prevent f(x)=0 for all x (=> To prevent rel error(f,f^{hat})=inf in Operator learning)
    for i in range(sum_num_f):
        f+=rand_f(is_rand_amp=True)
    f = f.to(torch.float32)

    #solve
    if alpha != 1.:
        f = (f * alpha).clamp(-10., 10.) # normalizer of ddpm is 10.
    return u0, f


def generate_data_burgers_equation(
        experiment: str,
        num_samples_train: int = 24000,
        # num_samples_valid: int = 0,
        num_samples_test: int = 6000,
        starting_time=0,
        end_time=1,
        nt=10, 
        nx=128, 
        device: torch.cuda.device = "cpu",
        varying_f = False, # generate data with Gaussian f varing in time.
        uniform = True, 
        partial_control = None, 
        alpha = 1, # distribution shift
        save_path = '', 
) -> None:

    """
    Generate data for Burgers' equation
    Args:
        experiment (str): experiment string
        pde (dict): dictionary for PDEs at different resolution
        num_samples (int): number of trajectories to solve
        device (torch.cuda.device): device (cpu/gpu)
        uniform: if True, generate Nf f for each u0, equivalently, every pair of f and u0 is evaluated
        varying_f: if True, f varies with time
        partial_control: if 'front_rear_quarter', nonzero f only at (0, 1/4) and (3/4, 1)
    Returns:
        None
    """
    print(f'Device: {device}')

    print(f'Equation: {experiment}')
    if uniform:
        print(f'Number of samples: {args.num_u0 * args.num_f}, uniform grid u and f')
        assert args.num_u0 * args.num_f == num_samples_train + num_samples_test, \
            'number of all samples should == Nu0 * Nf'
        assert not partial_control, 'only varying_f settings are supported for partial control'
    else:
        print(f'Number of samples: {num_samples_test + num_samples_train}, generating free u and f')
        assert varying_f, 'Only supports varying_f when every f is paired with a different u0'

    # Initial condition and time dependent force term
    if not uniform:
        u0, f = make_data_varying_f(
            Nu0=(num_samples_test + num_samples_train), 
            Nf=(num_samples_test + num_samples_train), 
            s=nx, 
            t=nt, 
            partial_control=partial_control, 
            alpha=alpha, 
        ) 
        u0 = torch.FloatTensor(u0).to(device)
        f = torch.FloatTensor(f).to(device)
        
        torch.cuda.synchronize()
        t1 = time.time()
        trajectory = burgers_numeric_solve_free(
            u0, f, visc=VISC, T=end_time - starting_time, dt=1e-4, 
            num_t=nt,
            mode='const' if not varying_f else None
        )
        torch.cuda.synchronize()
        t2 = time.time()
        print(f'Generation time: {t2 - t1:.4f}s')
        
        
    else:
        raise NotImplementedError("Not using the setting of Nf * Nu0 = Nsamples for now.")
    
    # Save solutions
    # zero pad to shape (N, 10, 128) and (N, 11, 128)
    
    index_range = range(trajectory.shape[0])   
    shuffled_indices = random.sample(index_range, len(index_range))  

    train_data = {
        'f': f[shuffled_indices[:num_samples_train]].cpu(), 
        'u': trajectory[shuffled_indices[:num_samples_train]].cpu()
    }
    test_data = {
        'f': f[shuffled_indices[num_samples_train:]].cpu(), 
        'u': trajectory[shuffled_indices[num_samples_train:]].cpu()
    }
    torch.save(train_data, save_path + 'train')
    torch.save(test_data, save_path + 'test')
    print("Data saved")



def log_datagen(args, file_dir=None):
    file_path = os.path.join(file_dir, 'log.yaml')
    with open(file_path, 'a+') as f:
        s = yaml.safe_load(f)
    # handle the case where s is empty
    if s is None:
        s = {'nothing': None}
    merge_save_dict(file_path, {args.seed: vars(args)})

def main(args):
    #gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)[-1]
    use_cuda = torch.cuda.is_available()
    print("Is available to use cuda? : ",use_cuda)

    check_directory(args.save_path)
    check_files(args.save_path)
    log_datagen(args, args.save_path)

    generate_data_burgers_equation(
        experiment=args.experiment,
        starting_time=args.start_time,
        end_time=args.end_time,
        num_samples_train=args.train_samples,
        # num_samples_valid=args.valid_samples,
        num_samples_test=args.test_samples,
        device=args.device, 
        nt = args.nt, 
        nx = args.nx, 
        varying_f=args.varying_f, 
        uniform=args.uniform_u_f, 
        partial_control=args.partial_control, 
        alpha=args.alpha, 
        save_path=args.save_path, 
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating PDE data')
    parser.add_argument('--experiment', type=str, default='burgers',
                        help='Experiment for which data should create for')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Used device')
    # sample number
    parser.add_argument('--num_f', default=1000, type=int,
                        help='the number of force data')
    parser.add_argument('--num_u0', default=100, type=int, 
                        help='the number of initial data')
    parser.add_argument('--train_samples', type=int, default=90000,
                        help='Samples in the training dataset')
    parser.add_argument('--test_samples', type=int, default=10000,
                        help='Samples in the test dataset')
    parser.add_argument('--log', type=eval, default=False,
                        help='pip the output to log file')
    # control mode
    parser.add_argument('--uniform_u_f', default=False, type=eval, 
                        help='Whether to use u \cross f to generate random \
                            samples. If False, num_f and num_u0 will not be used.')
    parser.add_argument('--varying_f', type=eval, default=True,
                        help='If the force sample varies over time')
    parser.add_argument('--partial_control', type=str, default=None,
                        help="If using partial control. Can be 'front_rear_quarter'")
    
    # simulation settings
    parser.add_argument('--nt', type=int, default=10,
                help='Time grids, (f has nt values over time and u is stamped nt+1 times, including u0)')
    parser.add_argument('--nx', type=int, default=128,
                help='Space grids.')
    
    parser.add_argument('--start_time', type=float, default=0.,
                help='Physical starting time')
    parser.add_argument('--end_time', type=float, default=1.,
                help='Physical ending time')
    # for ablation study. p(u|w) should work when the training set and the test set is different. Therefore, we generate different datasets with different alphas
    parser.add_argument('--alpha', type=float, default=1.,
                help='How much w is shifted from the original dataset')
    
    parser.add_argument('--save_path', type=str, default="data/free_u_f_1e5_seed0/",
                help='Which path to save the result into')
    
    parser.add_argument('--seed', type=int, default=0,
                help='Random seed')
    
    args = parser.parse_args()

    # use same seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 

    main(args)
