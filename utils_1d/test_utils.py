import copy
import sys
import torch
from dataset.data_1d import DiffusionDataset, get_burgers_preprocess
from diffusion.diffusion_1d import GaussianDiffusion, GaussianDiffusion1D
from utils_1d.train_diffusion import Trainer, Trainer1D
from model.model_1d.unet import Unet2D, Unet1D
from utils_1d.model_utils import get_nablaJ
from solver.burgers.burgers_solver import burgers_numeric_solve_free
from utils_1d.result_io import save_acc

RESCALER = 10.

def get_2d_ddpm(args):
    sim_time_stamps, sim_space_grids = 16, 128 

    if args.condition_on_residual is not None and args.condition_on_residual == 'residual_gradient':
        # If conditioned on residual, input dim becomes 4 (nabla_{u,w} residual)
        channels = 4
        if args.is_model_w:
            raise NotImplementedError
    elif args.is_model_w:
        if args.expand_condition:
            channels = 1
            channels += 1 if args.condition_u0 else 0
            channels += 1 if args.condition_uT else 0
        else:
            channels = 2
    else:
        channels = 3

    # make model
    if not args.eval_two_models:
        u_net = Unet2D(
            dim=args.dim, 
            init_dim = None,
            out_dim = 3,
            dim_mults=args.dim_muls,
            channels = channels, 
            self_condition = False,
            resnet_block_groups = args.resnet_block_groups,
            learned_variance = False,
            learned_sinusoidal_cond = False,
            random_fourier_features = False,
            learned_sinusoidal_dim = 16,
            sinusoidal_pos_emb_theta = 10000,
            attn_dim_head = 32,
            attn_heads = 4, 
            condition_on_residual = args.condition_on_residual, 
        )
    ddpm = GaussianDiffusion(
        u_net if not args.eval_two_models else (args.unet_uw, args.unet_w), 
        seq_length=(sim_time_stamps, sim_space_grids), 
        auto_normalize=False, 
        use_conv2d=True, 
        temporal=True, # using temporal conv (2d conv or 1d conv after 1d conv on space)
        is_condition_u0=args.is_condition_u0, 
        is_condition_uT=args.is_condition_uT, 
        is_condition_u0_zero_pred_noise=args.is_condition_u0_zero_pred_noise, 
        is_condition_uT_zero_pred_noise=args.is_condition_uT_zero_pred_noise, 
        train_on_partially_observed=args.train_on_partially_observed, 
        set_unobserved_to_zero_during_sampling=args.set_unobserved_to_zero_during_sampling, 
        conditioned_on_residual = args.condition_on_residual, 
        residual_on_u0 = args.residual_on_u0, 
        recurrence=args.recurrence, 
        recurrence_k=args.recurrence_k, 
        asynchronous = args.asynch_inference_mode,
        is_init_model = args.is_init_model,
        is_model_w=args.is_model_w, 
        eval_two_models=args.eval_two_models, 
        expand_condition=args.expand_condition, 
        prior_beta=args.prior_beta
    )
    return ddpm


def mse_deviation(u1, u2, partially_observed=None, report_all=False):
    if partially_observed is not None:
        u1, u2 = u1.clone(), u2.clone()
        if partially_observed == 'front_rear_quarter':
            Nx = u1.shape[-1]
            idx = torch.cat((torch.arange(0, Nx // 4), torch.arange((3 * Nx) // 4, Nx)))
            u1, u2 = u1[..., idx], u2[..., idx] # neglect center
    if report_all:
        mse = (u1 - u2).square().mean((-1, -2))
        mae = (u1 - u2).abs().mean((-1, -2))
        ep = 1e-5
        return mse, mae, mse / (u2 + ep).square().mean(), mae / (u2 + ep).abs().mean()
    return (u1 - u2).square().mean(-1)


def metric(
    u_target: torch.Tensor, 
    f: torch.Tensor, 
    target='final_u', 
    partial_control='full', 
    report_all=False, 
    diffused_u=None, 
    evaluate=False, 
    partially_observed=None, 
):
    '''
    Evaluates the control based on the state deviation and the control cost.
    Note that f and u should NOT be rescaled. (Should be directly input to the solver)

    Arguments:
        u_target:
            Ground truth states
            size: (batch_size, Nt, Nx) (currently Nt = 11, Nx = 128)
        f: 
            Generated control force
            size: (batch_size, Nt - 1, Nx) (currently Nt = 11, Nx = 128)
        eval: whether to calculate loss given diffused u. If true, evaluate how good the
            diffused u and f are.
    Returns:
        J_actual:
            Deviation of controlled u from target u for each sample, in MSE.
            When target is 'final_u', evaluate only at the final time stamp
            size: (batch_size)
        
        control_energy:
            Cost of the control force for each sample, in sum of square.
            size: (bacth_size)
    '''
    u_target, f = u_target.clone(), f.clone() # avoid side-effect

    assert len(u_target.size()) == len(f.size()) == 3
    # ensure partially control is correctly evaluated
    if type(partial_control) is None or partial_control == 'full':
        pass
    elif partial_control == 'front_rear_quarter':
        Nx = f.size(2)
        f[:, :, Nx // 4: (Nx * 3) // 4] = 0

    if evaluate:
        u_controlled = diffused_u.clone()
    else:
        u_controlled = burgers_numeric_solve_free(u_target[:, 0, :], f, visc=0.01, T=1.5, dt=1e-4, num_t=15)

    # partially observed: only evaluate the target on the observed region
    if partially_observed is not None:
        Nx = u_controlled.size(-1)
        if partially_observed == 'front_rear_quarter':
            idx = torch.cat((torch.arange(0, Nx // 4), torch.arange((3 * Nx) // 4, Nx)))
            u_controlled = u_controlled[..., idx]
            u_target = u_target[..., idx]
        else:
            raise NotImplementedError
    
    # eval J_actual
    if target == 'final_u':
        mse = (u_controlled[:, -1, :] - u_target[:, -1, :]).square().mean(-1) # MSE
        mse_median, _ = (u_controlled[:, -1, :] - u_target[:, -1, :]).square().median(-1)
        mae = (u_controlled[:, -1, :] - u_target[:, -1, :]).abs().mean(-1) # MAE
        mae_median, _ = (u_controlled[:, -1, :] - u_target[:, -1, :]).abs().median(-1)
        ep = 1e-5
        nmse = (u_controlled[:, -1, :] - u_target[:, -1, :]).square().mean(-1) / (u_target[:, -1, :].square().mean() + ep) # normalized MSE
        nmae = (u_controlled[:, -1, :] - u_target[:, -1, :]).abs().mean(-1) / (u_target[:, -1, :].abs().mean() + ep) # normalized MAE

        if not report_all:
            J_actual = mse
        else:
            J_actual = (mse, mse_median, mae, mae_median, nmse, nmae)
    else:
        raise ValueError('Undefined target to evaluate')
    
    control_energy = f.square().sum((-1, -2))

    return J_actual, control_energy

def mse_dist_reg(u):
    return (u[:, 1:, :] - u[:, :-1, :]).square().sum() 

def ddpm_guidance_loss(
        u_target, u, f, 
        wu=0, wf=0, wreg=0, wpinn=0,  
        dist_reg=lambda x: 0, 
        pinn_loss_mode='mean', 
        partially_observed=None
):
    '''
    Arguments:
        u_target: (batch_size, Nt, Nx)
        u: (batch_size, Nt, Nx)
        f: (batch_size, Nt - 1, Nx)
        
    '''

    u0_gt = u_target[:, 0, :]
    uf_gt = u_target[:, -1, :]

    u0 = u[:, 0, :]
    uf = u[:, -1, :]

    loss_u = (u-uf_gt.unsqueeze(1).expand(-1,u.shape[1],-1)).square()
    
    if partially_observed is not None:
        if partially_observed == 'front_rear_quarter':
            nx = u.shape[-1]
            loss_u[:, nx // 4: (nx * 3) // 4] = 0
        else:
            raise ValueError('Unknown partially observed mode')
    loss_u = loss_u.mean()

    loss_f = f.square().sum((-1, -2)).mean()

    if wpinn != 0:
        loss_pinn = pinn_loss(u, f, mode=pinn_loss_mode, partially_observed=partially_observed)
    else:
        loss_pinn = 0

    return loss_u * wu + loss_f * wf + loss_pinn * wpinn + dist_reg(u) * wreg



# Loading dataset and model

def load_burgers_dataset(dataset):
    tmp_dataset = DiffusionDataset(
        dataset, # dataset of f varying in both space and time
        preprocess=get_burgers_preprocess(
            rescaler=RESCALER, 
            stack_u_and_f=True, 
            pad_for_2d_conv=True, 
            partially_observed_fill_zero_unobserved = None, # does not matter since only for loading models
        )
    )
    return tmp_dataset


def get_target(target_i, f=False, device=0, dataset='free_u_f_1e5', **dataset_kwargs):
    # repeating in the first dimension if one target is shared with multiple f
    test_dataset = DiffusionDataset(
        dataset, # dataset of f varying in both space and time
        preprocess=get_burgers_preprocess(
            rescaler=1., 
            stack_u_and_f=False, 
            pad_for_2d_conv=False, 
            **dataset_kwargs, 
        )
    )

    if not f: # return only u
        ret = test_dataset.get(target_i).cuda(device)[..., :16, :]
    else: # return only f
        ret = test_dataset.get(target_i).cuda(device)[..., 16:, :]

    if len(ret.size()) == 2: # if target_i is int
        ret = ret.unsqueeze(0)

    return ret


def use_args_w(args):
    args = copy.deepcopy(args)
    model_w_key = '__model_w'
    for k in args.__dict__.keys():
        if model_w_key in k:
            setattr(args, k[:-len(model_w_key)], getattr(args, k))
    
    return args


def load_2dconv_model_two_ddpm(i, args):
    args = copy.deepcopy(args) # was a bug... should not modify the arg used in the outer scope...
    dataset = load_burgers_dataset(args.dataset)
    args.is_ddpm_w = False
    args.eval_two_models = False # when loading the separately trained model, should not use eval_two_models
    ddpm_uw = get_2d_ddpm(args)
    trainer = Trainer(
        ddpm_uw, 
        dataset, 
        results_folder=f'./trained_models/burgers/{args.exp_id}/', 
        train_num_steps=args.train_num_steps, 
        save_and_sample_every=args.checkpoint_interval, 
    )
    trainer.load(args.checkpoint if 'checkpoint' in args.__dict__ else 10)
    unet_uw = ddpm_uw.model

    # load p(w)
    args.is_ddpm_w = True
    args.eval_two_models = False
    args = use_args_w(args)
    ddpm_w = get_2d_ddpm(args)
    trainer = Trainer(
        ddpm_w, 
        dataset, 
        results_folder=f'./trained_models/burgers_w/{args.exp_id__model_w}/', 
        train_num_steps=args.train_num_steps, 
        save_and_sample_every=args.checkpoint_interval, 
    )
    trainer.load(args.checkpoint if 'checkpoint' in args.__dict__ else 10)
    unet_w = ddpm_w.model

    args.eval_two_models = True
    args.is_ddpm_w = False
    args.unet_uw = unet_uw
    args.unet_w = unet_w
    ddpm_two_models = get_2d_ddpm(args)
    return ddpm_two_models.cuda()
    
    
    
def load_2dconv_model(i, args, new=True):
    if args.eval_two_models:
        assert not args.is_model_w
        return load_2dconv_model_two_ddpm(i, args)
    elif args.is_model_w:
        return load_2dconv_model_w(args)
    
    dataset = load_burgers_dataset(args.dataset)
    ddpm = get_2d_ddpm(args)

    trainer = Trainer(
        ddpm, 
        dataset, 
        results_folder=f"{args.diffusion_model_path}/{i}/", 
        train_num_steps=args.train_num_steps, 
        save_and_sample_every=args.checkpoint_interval, 
    )
    trainer.load(args.checkpoint if 'checkpoint' in args.__dict__ else 10)
    return ddpm


def load_2dconv_model_w(args):
    # copy args for loading model w
    args = use_args_w(copy.deepcopy(args))
    dataset = load_burgers_dataset(args.dataset)
    ddpm = get_2d_ddpm(args)

    trainer = Trainer(
        ddpm, 
        dataset, 
        results_folder=f"{args.diffusion_model_path}/{args.exp_id__model_w}/", 
        train_num_steps=args.train_num_steps__model_w, 
        save_and_sample_every=args.checkpoint__model_w, 
    )
    trainer.load(args.checkpoint__model_w)
    return ddpm

