import argparse
import datetime
import matplotlib.pylab as plt
import numpy as np
import pdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import tqdm
from tqdm.auto import tqdm
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from dataset.data_2d import Smoke
from diffusion.diffusion_2d import GaussianDiffusion, Trainer, Simulator
from dataset.apps.evaluate_solver import *
from model.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D

class solver_env():
    def __init__(self):
        super().__init__()
        self.smoke_out_value_t = None
        self.smoke_outs = None 

    def solver_reset(self, data):
        """        
        Output:
            density: numpy array, [64, 64]
            velocity: numpy array, [128, 128, 2]
        """
        
        init_velocity = data.copy()[:,0,1:3].transpose(0,2,3,1) # 1, 64, 64, 2
        init_density = data.copy()[0,0,0,:,:] # nx, nx 
        self.smoke_outs = np.zeros((7,), dtype=float) 
        return init_velocity[0], init_density

    def solver_step(self, sim, density_t, velocity_t, c1_t, c2_t, t, bucket_index=1):
        """
        Input: 
            sim: environment of the fluid
            density_t: numpy array, [64, 64]
            velocity_t: numpy array, [64, 64, 2]
            c1_t: numpy array, [64, 64]
            c2_t: numpy array, [64, 64]
            bucket_index: int, the index of the target bucket
        
        Output:
            density: numpy array, [64, 64]
            velocity: numpy array, [64, 64, 2]
            smoke_out_value_t: numpy array, [1], the ratio of out from the target bucket
            smoke_out: numpy array, [7].
        """
        
        velocity_t = np.expand_dims(velocity_t,axis=0)
        c1_t = np.expand_dims(c1_t, axis=0)
        c2_t = np.expand_dims(c2_t, axis=0)
        if t == 0:
            density_t, zero_densitys_t, velocity_t, smoke_out_value_t, self.smoke_outs = solver(sim, velocity_t, density_t, c1_t, c2_t, t, self.smoke_outs, per_timelength=1, bucket_index=bucket_index)
        else: 
            density_t, zero_densitys_t, velocity_t, smoke_out_value_t, self.smoke_outs = solver(sim, velocity_t, density_t, c1_t, c2_t, t, self.smoke_outs, per_timelength=1, bucket_index=bucket_index)
            
        density_t = density_t[-1]
        zero_densitys_t = zero_densitys_t[-1]
        velocity_t = velocity_t[-1]        

        return density_t, zero_densitys_t, velocity_t, smoke_out_value_t


def guidance_fn(x, RESCALER, w_energy=0):
    '''
    low, init: rescaled
    init_u: not rescaled
    '''
    x = x * RESCALER # [batch, T, 6, 64, 64]

    state = x
    guidance_success = state[:,-1,-1].mean((-1,-2)).sum()
    guidance_energy = state[:,:,3:5].square().mean((1,2,3,4)).sum()
    guidance = -guidance_success + w_energy * guidance_energy
    grad_x = grad(guidance, x, grad_outputs=torch.ones_like(guidance))[0]

    return grad_x

def create_model(args):
    model = Unet3D_with_Conv3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels=6
    )
    
    print("Number of parameters: {}".
          format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    return model

def load_ddpm_model(args, shape, ori_shape, RESCALER):
    if args.asynch_inference_mode:
        assert args.infer_interval == 1
    assert args.infer_interval < args.horizon
    
    model_synch = create_model(args)
    model_synch.to(args.device)
    diffusion_synch = GaussianDiffusion(
        model_synch,
        RESCALER,
        args.is_condition_control,
        args.is_condition_pad,
        ori_shape,
        image_size = args.image_size,
        horizon=args.horizon,
        diffusion_steps = args.diffusion_steps,
        sampling_timesteps=args.ddim_sampling_steps if args.using_ddim else args.diffusion_steps, # ddim, accelerate sampling
        ddim_sampling_eta=args.ddim_eta,
        loss_type = 'l2',            # L1 or L2
        objective = 'pred_noise',
        standard_fixed_ratio = args.standard_fixed_ratio, # used in standard sampling
        coeff_ratio = args.coeff_ratio, # used in standard-alpha sampling
        asynch_inference_mode = args.asynch_inference_mode, # this affects number of denosing steps and returned data format
        is_synch_model = True
    )
    diffusion_synch.eval()
    # load trainer
    trainer_synch = Trainer(
        diffusion_synch,
        dataset = args.dataset,
        dataset_path = args.dataset_path,
        horizon = args.horizon,
        results_path = args.init_diffusion_model_path, 
    )
    trainer_synch.load(args.init_diffusion_checkpoint) 

    if args.asynch_inference_mode:
        assert args.infer_interval == 1
        model_asynch = create_model(args)
        model_asynch.to(args.device)
        diffusion_asynch = GaussianDiffusion(
            model_asynch,
            RESCALER,
            args.is_condition_control,
            args.is_condition_pad,
            ori_shape,
            image_size = args.image_size,
            horizon = args.horizon, 
            diffusion_steps = args.diffusion_steps,
            sampling_timesteps=args.ddim_sampling_steps if args.using_ddim else args.diffusion_steps, # ddim, accelerate sampling
            ddim_sampling_eta=args.ddim_eta,
            loss_type = 'l2',            # L1 or L2
            objective = 'pred_noise',
            standard_fixed_ratio = args.standard_fixed_ratio, # used in standard sampling
            coeff_ratio = args.coeff_ratio, # used in standard-alpha sampling
            asynch_inference_mode = True,
            is_synch_model = False # asyn denosing, denosing steps=diffusion_steps/horizon
        )
        diffusion_asynch.eval()
        # load trainer
        trainer_asynch = Trainer(
            diffusion_asynch,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            horizon=args.horizon,
            results_path = args.online_diffusion_model_path, # diffuse on both cond states, pred states and boundary
        )
        trainer_asynch.load(args.online_diffusion_checkpoint)
    model = [diffusion_synch, diffusion_asynch] if args.asynch_inference_mode else [diffusion_synch]

    return model, trainer_synch.device

def load_model(args, shape, ori_shape, RESCALER, w_energy=0):
    # load main model of each inference method
    if args.inference_method == 'DDPM':
        model, device = load_ddpm_model(args, shape, ori_shape, RESCALER)
        RESCALER = RESCALER.to(device)
    else:
        assert False, "Not implemented yet"
    
    # define design function
    def design_fn(x):
        grad_x = guidance_fn(x, RESCALER, w_energy=w_energy)
    
        return grad_x
    
    if args.inference_method == 'DDPM':
        return model, design_fn
    else:
        assert False, "Not implemented yet"
     
class InferencePipeline(object):
    def __init__(
        self,
        model,
        args=None,
        RESCALER=1,
        results_path=None,
        args_general=None,
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.results_path = results_path
        self.args_general = args_general
        self.is_condition_control = args_general.is_condition_control
        self.image_size = self.args_general.image_size
        self.device = self.args_general.device
        self.upsample = args_general.upsample
        self.total_time_steps = self.args_general.total_time_steps
        self.bucket_index = self.args_general.bucket_index
        self.rand_action_p = self.args_general.rand_action_p
        self.rand_action_std = self.args_general.rand_action_std
        self.random_sim = self.args_general.random_sim
        if self.random_sim: # different env (locations of inner obstacles) for different sim_id
            self.random_obstacle_array = np.load(os.path.join(self.args_general.dataset_path, "test", "random_obstacle.npy")) # shape: [100, 5, 2]
        self.RESCALER = RESCALER.to(self.device)

        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        if not os.path.exists(os.path.join(self.results_path, "densitys")):
            os.makedirs(os.path.join(self.results_path, "densitys"))
        if not os.path.exists(os.path.join(self.results_path, "velocities")):
            os.makedirs(os.path.join(self.results_path, "velocities"))
        if not os.path.exists(os.path.join(self.results_path, "controls")):
            os.makedirs(os.path.join(self.results_path, "controls"))
        if not os.path.exists(os.path.join(self.results_path, "smoke_outs")):
            os.makedirs(os.path.join(self.results_path, "smoke_outs"))

    def init_sim_env(self, random_obstacle=None):
        return init_sim(random_obstacle)
    

    def parse_init_state(self, state, t=0, unnormalize=True, rand_action_p=0, std=1, x_upper=10, x_lower=-10, y_upper=10, y_lower=-2):
        if unnormalize:
            state = state * self.RESCALER
        c1_t = state[0,t,3].cpu().numpy()
        c2_t = state[0,t,4].cpu().numpy()
        if np.random.rand() < rand_action_p:
            c1_t = np.random.uniform(x_lower, x_upper, c1_t.shape)
            c2_t = np.random.uniform(y_lower, y_upper, c2_t.shape)
        return c1_t, c2_t

    def update_init_conditions(self, pred, density_t, velocity_t, smoke_out_value_t, normalize=True):
        density_t = torch.tensor(density_t).to(self.args_general.device).unsqueeze(0) # output [1, 64, 64]
        velocity_t = torch.tensor(velocity_t).permute(2, 0, 1).unsqueeze(0).to(self.args_general.device) # output [1, 2, 64, 64]
        smoke_out_value_t = torch.tensor(smoke_out_value_t[0]).reshape(1, 1, 1).expand(1, self.image_size, self.image_size).to(self.args_general.device) # output [1, 1, 64, 64]
        if normalize:
            density_t = density_t / self.RESCALER[0,0,0,0,0]
            velocity_t = velocity_t / self.RESCALER[0,0,1:3]
            smoke_out_value_t = smoke_out_value_t / self.RESCALER[0,0,5,0,0]
        if args.asynch_inference_mode: 
            pred = pred[:, 1:]
        pred[:, 0, 0] = density_t
        pred[:, 0, 1:3] = velocity_t
        pred[:, 0, 5] = smoke_out_value_t

        return pred
    
    def run_model_DDPM(self, state):
        '''
        state: not rescaled
        '''
        def anti_diagonal(pred):
            """
            pred: List[Tensor(b, 6, 64, 64)]
            """
            return torch.cat([pred[len(pred) - 1 - t][:,t].unsqueeze(1) for t in range(len(pred))], dim=1)
        
        diffusion_synch = self.model[0]
        
        env = solver_env()
        
        velocity_t, density_t = env.solver_reset(state.detach().cpu().numpy()[:,:,:,:,:])  # [128,128,2], [64,64]        
        state = state.to(self.args_general.device) # [b, 65, 6, 64, 64]
        density, velocity, control, smoke_out = [density_t], [velocity_t], [], []
        if args.asynch_inference_mode: # our method: CL-DiffPhyCon
            assert args.infer_interval == 1
            assert len(self.model) == 2
            diffusion_asynch = self.model[1] 
            # initialization by anti-diagonal results of diffusion_synch model
            print("start initialization of online inference...")
            pred = diffusion_synch.sample(
                batch_size = state.shape[0],
                design_fn=self.args["design_fn"],
                design_guidance=self.args["design_guidance"],
                init=state /self.RESCALER
            )
            pred = anti_diagonal(pred) # output: [b, 10, 6, 64, 64]
            print("initialization of online inference finished.")
            
            for t in tqdm(range(0, self.total_time_steps), desc = 'online time step'):
                pred = diffusion_asynch.sample(
                    batch_size = state.shape[0],
                    design_fn=self.args["design_fn"],
                    design_guidance=self.args["design_guidance"],
                    init=pred # to be replaced by real ones
                )
                c1_t, c2_t = self.parse_init_state(pred, t=0, unnormalize=True, rand_action_p=self.rand_action_p, std=self.rand_action_std)
                density_t, zero_densitys_t, velocity_t, smoke_out_value_t = env.solver_step(self.sim, density_t, velocity_t, c1_t, c2_t, t, bucket_index=self.bucket_index)
                pred = self.update_init_conditions(pred, density_t, velocity_t, smoke_out_value_t, normalize=True)
                density.append(density_t)
                velocity.append(velocity_t)
                control.append(np.array([c1_t, c2_t]))
                smoke_out.append(smoke_out_value_t)
            return np.stack(density), np.stack(velocity), np.stack(control), np.stack(smoke_out)
        else: # baseline: synch DDPM, i.e., DiffPhyCon-$h$ in the paper, where $h$=infer_interval
            assert len(self.model) == 1
            for t in range(0, self.total_time_steps):
                print("online step: ", t)
                t_ = t % args.infer_interval
                if t_ == 0:
                    print("make new round of denoising from total noise")
                    pred = diffusion_synch.sample(
                        batch_size = state.shape[0],
                        design_fn=self.args["design_fn"],
                        design_guidance=self.args["design_guidance"],
                        init= state / self.RESCALER if t == 0 else pred
                    )
                c1_t, c2_t = self.parse_init_state(pred, t_, unnormalize=True, rand_action_p=self.rand_action_p, std=self.rand_action_std)
                density_t, zero_densitys_t, velocity_t, smoke_out_value_t = env.solver_step(self.sim, density_t, velocity_t, c1_t, c2_t, t, bucket_index=self.bucket_index)
                if t_ == args.infer_interval - 1:
                    print("update condition")
                    pred = self.update_init_conditions(pred, density_t, velocity_t, smoke_out_value_t, normalize=True)
                density.append(density_t)
                velocity.append(velocity_t)
                control.append(np.array([c1_t, c2_t]))
                smoke_out.append(smoke_out_value_t)

            return np.stack(density), np.stack(velocity), np.stack(control), np.stack(smoke_out)

    def run(self, dataloader):
        preds = []
        for i, data in enumerate(dataloader):
            print(f"Batch No.{i}")
            state, shape, ori_shape, sim_id = data
            """
            Initialize simulator environment regardless of sim_id
            """
            start_time = time.time()
            random_obstacle = self.random_obstacle_array[i] if self.random_sim else None
            self.sim = self.init_sim_env(random_obstacle)

            if self.args_general.inference_method == "DDPM":
                pred = self.run_model_DDPM(state)
                densitys, velocities, controls, smoke_outs = pred # density: [32, 64, 64], velocity: [32, 128, 128, 2], control: [31, 64, 64], smoke_out: [31, 8]
            else:
                assert False, "Not implemented yet"
            end_time = time.time()
            print("time cost: ", end_time - start_time)
            preds.append(pred)

            self.save(sim_id, pred, self.results_path) 
    
    def save(self, sim_id, pred, results_path):
        densitys, velocities, controls, smoke_outs = pred
        for index in range(sim_id.shape[0]):
            id = sim_id[index].cpu().item()
            densitys_filepath = os.path.join(results_path, "densitys", "{}.npy".format(id))
            velocities_filepath = os.path.join(results_path, "velocities", "{}.npy".format(id))
            controls_filepath = os.path.join(results_path, "controls", "{}.npy".format(id))
            smoke_outs_filepath = os.path.join(results_path, "smoke_outs", "{}.npy".format(id))
            with open(densitys_filepath, 'wb') as f:
                np.save(f, densitys)
            with open(velocities_filepath, 'wb') as f:
                np.save(f, velocities)
            with open(controls_filepath, 'wb') as f:
                np.save(f, controls)
            with open(smoke_outs_filepath, 'wb') as f:
                np.save(f, smoke_outs)
            print("id {} saved at {}: ".format(id, results_path))


def inference(dataloader, model, design_fn, args, RESCALER):
    model_args = {
        "design_fn": design_fn,
        "design_guidance": args.design_guidance,
    }

    inferencePPL = InferencePipeline(
        model, 
        model_args,
        RESCALER,
        results_path = args.inference_result_subpath,
        args_general=args
    )
    inferencePPL.run(dataloader)
    

def load_data(args):
    if args.dataset == "Smoke":
        dataset = Smoke(
            dataset_path=args.dataset_path,
            is_train=True,
        )
        _, shape, ori_shape, _ = dataset[0]
    else:
        assert False
    RESCALER = dataset.RESCALER.unsqueeze(0).to(args.device)

    dataset = Smoke(
        dataset_path=args.dataset_path,
        is_train=False,
    )
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = 32)
    print("number of batch in test_loader: ", len(test_loader))
    return test_loader, shape, ori_shape, RESCALER

def main(args):
    dataloader, shape, ori_shape, RESCALER = load_data(args)
    diffusion, design_fn = load_model(args, shape, ori_shape, RESCALER, args.w_energy)
    inference(dataloader, diffusion, design_fn, args, RESCALER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference 2d inverse design model')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--dataset', default='Smoke', type=str,
                        help='dataset to evaluate')
    parser.add_argument('--dataset_path', default="/data/", type=str,
                        help='path to dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id used in training')
    parser.add_argument('--w_energy', default=0, type=float,
                        help='guidance intensity of initial condition')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--upsample', default=0, type=int,
                        help='number of times of upsampling with super resolution model, n *= 2**upsample')
    parser.add_argument('--is_condition_control', default=False, type=eval,
                        help='If condition on control')
    parser.add_argument('--is_condition_pad', default=True, type=eval,
                        help='If condition on padded state')
    parser.add_argument('--is_condition_reward', default=False, type=eval,
                        help='If condition on padded state')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='size of batch of input to use')
    parser.add_argument('--inference_result_path', default="/home/diffusion_operator/2d/results/test/", type=str,
                        help='path to save inference result')
    parser.add_argument('--inference_method', default="DDPM", type=str,
                        help='the inference method: DDPM')
    parser.add_argument('--bucket_index', default=1, type=int, help='bucket index for smoke_out, choices: [0, 1, 2, 3, 4, 5, 6, 7], the top middle bucket is 1')
    parser.add_argument('--rand_action_p', type=float, default=0, help='probability of random action')
    parser.add_argument('--random_sim', action='store_true', help='if true, use random obstacle map for simulation, i.e., OOD')
    parser.add_argument('--rand_action_std', type=float, default=1, help='std of random action')

    # DDPM
    parser.add_argument('--horizon', default=15, type=int,
                        help='number of time steps inside a window of diffusion models, i.e., $H$ in paper')
    parser.add_argument('--diffusion_steps', default=600, type=int,
                        help='number of denoising steps in diffusion model, i.e., $T$ in paper')
    parser.add_argument('--diffusion_model_path', default="/data/results/", type=str,
                        help='directory of trained diffusion model (Unet)')
    parser.add_argument('--diffusion_checkpoint', default=50, type=int,
                        help='index of checkpoint of trained diffusion model (Unet)')
    parser.add_argument('--using_ddim', default=False, type=eval,
                        help='If using DDIM')
    parser.add_argument('--ddim_eta', default=0.3, type=float, help='$eta$ in DDIM paper')
    parser.add_argument('--ddim_sampling_steps', default=75, type=int, 
                        help='DDIM sampling steps. Should be a divisor of diffusion_steps and also a multiple of horizon')
    parser.add_argument('--design_guidance', default='standard', type=str,
                        help='design_guidance', choices=['standard', 'standard-fixed', 'standard-alpha', 'universal-backward'])
    parser.add_argument('--standard_fixed_ratio', default=1000, type=float,
                        help='fixed ratio for standard-fixed in design_guidance sampling')
    parser.add_argument('--coeff_ratio', nargs='+', default=0, type=float,
                        help='coeff_ratio for standard-alpha sampling')
    parser.add_argument('--init_diffusion_model_path', default="/root/user/data/control/checkpoints/diffusion_theta_dim_mults_1_2_4/", type=str,
                        help='directory of trained diffusion model (Unet) for initialization of close loop control')
    parser.add_argument('--online_diffusion_model_path', default="/root/user/data/control/checkpoints/diffusion_theta_dim_mults_1_2_4/", type=str,
                        help='directory of trained diffusion model (Unet) for online asynchronous inference (after initialization) of close loop control')
    parser.add_argument('--init_diffusion_checkpoint', default=15, type=int,
                        help='index of checkpoint of trained diffusion model for initialization of close loop control')
    parser.add_argument('--online_diffusion_checkpoint', default=15, type=int,
                        help='index of checkpoint of trained diffusion model for online asynchronous inference (after initialization) of close loop control')
    parser.add_argument('--asynch_inference_mode', action='store_true', 
                    help="if true, all time steps are denoised asynchronously (our method), and only infer_interval=1 allowed; otherwise synchronously (baseline) for inference, and 1<=infer_interval<T")
    parser.add_argument('--infer_interval', type=int, default=1, help='interval of physical time steps for doing new denoising, i.e., $h$ in paper. \
                        Should be smaller than horizon. And should be 1 if asynch_inference_mode is True')
    parser.add_argument('--total_time_steps', type=int, default=64, help='number of total time steps of online control, i.e., $N$ in paper')

    # get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    # check validity of ddim_sampling_steps  
    if args.using_ddim:
        assert args.diffusion_steps % args.ddim_sampling_steps == 0 and args.ddim_sampling_steps % args.horizon == 0, "ddim_sampling_steps should be a divisor of diffusion_steps and a multiple of horizon"        
    
    if True:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        head = "unet_"
        method_label = "_syn_intervel_{}".format(args.infer_interval) if not args.asynch_inference_mode else "_asyn"
        ddim_lable = "_ddim_eta_{}_sampling_steps_{}".format(args.ddim_eta, args.ddim_sampling_steps) if args.using_ddim else ""
        rand_action_p_label = "_uniform_rap_{}".format(args.rand_action_p) if args.rand_action_p > 0 else ""
        if args.design_guidance == "standard":
            guid_ratio_label = "_sfr_{}".format(args.standard_fixed_ratio)
        elif args.design_guidance == "standard-alpha":
            guid_ratio_label = "_cr_{}".format(args.coeff_ratio)
        else:
            guid_ratio_label = ""
        random_sim_lable = "_randsim" if args.random_sim else ""
        args.inference_result_subpath = os.path.join(
            args.inference_result_path,
            head +
            current_time + 
            method_label + 
            ddim_lable + 
            rand_action_p_label +
            random_sim_lable +
            guid_ratio_label + 
            "_ckpt_{}".format(args.init_diffusion_checkpoint) +
            "_hrz_{}".format(args.horizon) +
            "_dstep_{}".format(args.diffusion_steps)
        )
        print("args: ", args)
        main(args)