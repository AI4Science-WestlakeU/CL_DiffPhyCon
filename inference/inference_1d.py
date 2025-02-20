import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import copy
import torch
import numpy as np
from utils_1d.model_utils import cosine_beta_J_schedule, get_nablaJ, plain_cosine_schedule, sigmoid_schedule, sigmoid_schedule_flip
from solver.burgers.burgers_solver import burgers_numeric_solve_free

from utils_1d.result_io import save_acc
from utils_1d.test_utils import RESCALER, ddpm_guidance_loss, mse_deviation, mse_dist_reg
from utils_1d.test_utils import metric, get_target, load_2dconv_model
import argparse
from utils_1d.utils import none_or_str
import time

parser = argparse.ArgumentParser(description='Eval model')
parser.add_argument('--exp_id_i', type=str,
                    help='trained model id of init')
parser.add_argument('--exp_id_f', type=str,
                    help='trained model id of feedback')
parser.add_argument('--model_str_in_key', default='', type=str,
                    help='description that appears in the result dict key')
parser.add_argument('--save_file', default='burgers_results/result_zerowf.yaml', type=str,
                    help='file to save')
parser.add_argument('--eval_save', default='eval', type=str,
                    help='file to save')
parser.add_argument('--dataset', default='free_u_f_1e5_seed0', type=str,
                    help='dataset name for evaluation (eval samples drawn from)')
parser.add_argument('--model_str', default='', type=str,
                    help='description (where is this used?)')

# experiment settings
parser.add_argument('--partial_control', default='full', type=none_or_str,
                    help='partial control setting. full or None if fully controlled')
parser.add_argument('--partially_observed', default=None, type=none_or_str, 
                    help='If None, fully observed, otherwise, partially observed during training\
                        Note that the force is always fuly observed. Possible choices:\
                        front_rear_quarter')
parser.add_argument('--train_on_partially_observed', default=None, type=none_or_str, 
                    help='Whether to train the model to generate zero states at the unobserved locations.')
parser.add_argument('--set_unobserved_to_zero_during_sampling', default=False, type=eval, 
                    help='Set central 1/2 to zero in each p sample loop.')

# p(u, w) model training
parser.add_argument('--checkpoint', default=10, type=int,
                    help='which checkpoint model to load')
parser.add_argument('--checkpoint_interval', default=10000, type=int,
                    help='save checkpoint every checkpoint_interval steps')
parser.add_argument('--train_num_steps', default=100000, type=int,
                    help='train_num_steps')

# sampling 
parser.add_argument('--J_scheduler', default=None, type=str,
                    help='which J_scheduler to use. None means no scheduling.')
parser.add_argument('--recurrence', default=False, type=eval, help='whether to use recurrence in Universal Guidance for Diffusion Models')
parser.add_argument('--recurrence_k', default=1, type=int, help='how many iterations of recurrence. k in Algo 1 in Universal Guidance for Diffusion Models')
parser.add_argument('--wfs', nargs='+', default=[0], type=float,
                    help='guidance intensity of energy')
parser.add_argument('--wus', nargs='+', default=[0], type=float,
                    help='guidance intensity of state deviation')
parser.add_argument('--wreg', default=0, type=float,
                    help='guidance intensity of proximity regularization')
parser.add_argument('--wpinns', nargs='+', default=[0], type=float,
                    help='guidance intensity of state deviation')
parser.add_argument('--pinn_loss_mode', default='mean', type=str,
                    help="Mode of PINN loss evaluation. Choose from ('mean', 'forward', 'backward)")
# residual
parser.add_argument('--condition_on_residual', default=None, type=str, 
                    help='option: None, residual_gradient')
parser.add_argument('--residual_on_u0', default=False, type=eval, 
                    help='when using conditioning on residual, whether feeding u0 or ut into Unet')

# ddpm and unet
parser.add_argument('--is_condition_u0', default=False, type=eval,
                    help='If learning p(u_[1, T-1] | u0)')
parser.add_argument('--is_condition_uT', default=False, type=eval,
                    help='If learning p(u_[0, T-1] | uT)')
parser.add_argument('--is_condition_u0_zero_pred_noise', default=True, type=eval,
                    help='If enforcing the pred_noise to be zero for the conditioned data\
                         when learning p(u_[1, T-1] | u0). if false, mimic bad behavior of exp 0 and 1')
parser.add_argument('--is_condition_uT_zero_pred_noise', default=True, type=eval,
                    help='If enforcing the pred_noise to be zero for the conditioned data\
                         when learning p(u_[1, T-1] | u0). if false, mimic bad behavior of exp 0 and 1')

# unet hyperparam
parser.add_argument('--dim', default=64, type=int,
                    help='first layer feature dim num in Unet')
parser.add_argument('--resnet_block_groups', default=1, type=int,
                    help='')
parser.add_argument('--dim_muls', nargs='+', default=[1, 2, 4, 8], type=int,
                    help='dimension of channels, multiplied to the base dim\
                        seq_length % (2 ** len(dim_muls)) must be 0')

# two ddpm: learn p(w, u) and p(w) -> use p(u | w) during inference
parser.add_argument('--is_model_w', default=False, type=eval, help='If training the p(w) model, else train the p(u, w) model')
parser.add_argument('--eval_two_models', default=False, type=eval, help='Set to False in this training file')
parser.add_argument('--expand_condition', default=False, type=eval, help='Expand conditioning information of u0 or uT in separate channels')
parser.add_argument('--prior_beta', default=1, type=float, help='strength of the prior (1 is p(u,w); 0 is p(u|w))')
parser.add_argument('--normalize_beta', default=False, type=eval, help='')
parser.add_argument('--w_scheduler', default=None, type=none_or_str,
                    help='which scheduler to use in front of nabla log p(w). None means no scheduling.')
parser.add_argument('--exp_id__model_w', type=str,
                    help='trained model id')
# training
parser.add_argument('--checkpoint__model_w', default=10, type=int,
                    help='which checkpoint model to load')
parser.add_argument('--checkpoint_interval__model_w', default=10000, type=int,
                    help='save checkpoint every checkpoint_interval steps')
parser.add_argument('--train_num_steps__model_w', default=100000, type=int,
                    help='train_num_steps')
# unet
parser.add_argument('--dim__model_w', default=64, type=int,
                    help='first layer feature dim num in Unet')
parser.add_argument('--resnet_block_groups__model_w', default=1, type=int,
                    help='')
parser.add_argument('--dim_muls__model_w', nargs='+', default=[1, 2, 4, 8], type=int,
                    help='dimension of channels, multiplied to the base dim\
                        seq_length % (2 ** len(dim_muls)) must be 0')
parser.add_argument('--infer_interval', default=1, type=int, 
                    help="synch feedback time gap.")
parser.add_argument('--asynch_inference_mode', action='store_true', 
                    help="if true, all time steps are denoised asynchronously (our method), and only infer_interval=1 allowed; otherwise synchronously (baseline) for inference, and 1<=infer_interval<T")
parser.add_argument('--is_init_model', default=True, type=bool, 
                    help="if true, the original DDPM, if False, ours new.")

parser.add_argument('--diffusion_model_path', default="/usr", type=str,
                        help='directory of trained diffusion model (Unet) for initialization of close loop control')

# loss, guidance and utils
def get_loss_fn_2dconv(
        wf=0, wu=0, wpinn=0, target_i=0, 
        wu_eval=1, wf_eval=0, device=0, 
        dataset='free_u_f_1e5', 
        dist_reg=lambda x: 0, 
        wreg=0, 
        partially_observed=None, 
        pinn_loss_mode='mean', 
):
    u_target = get_target(
        target_i, 
        device=device, 
        dataset=dataset, 
        partially_observed_fill_zero_unobserved=partially_observed
    )
    def loss_fn_2dconv(x, eval=False):
        if eval:
            raise NotImplementedError('Should not have used this branch. Should report loss also using the custom metric function')
            return ddpm_guidance_loss(
                u_target, x[:, 0, :16, :], x[:, 1, :15, :], 
                wu=wu_eval, 
                wf=wf_eval, 
                partially_observed=None
            )
        else:
            # use rescaled value in guidance
            return ddpm_guidance_loss(
                u_target / RESCALER, x[:, 0, :16, :], x[:, 1, :15, :], 
                wu=wu, wf=wf, wpinn=wpinn, 
                dist_reg=dist_reg, 
                pinn_loss_mode=pinn_loss_mode, 
                wreg=wreg, 
                partially_observed=partially_observed, # only calculate guidance on the observed locations
            )
    return loss_fn_2dconv

def get_nablaJ_2dconv(**kwargs):
    return get_nablaJ(get_loss_fn_2dconv(**kwargs))


# run exp
def diffuse_2dconv(args, custom_metric, model_i, model_f, seed=0, ret_ls=False, **kwargs):
    
    def anti_diagonal(pred):  
        return torch.cat([pred[len(pred) - 1 - t][:,t].unsqueeze(1) for t in range(len(pred))], dim=1)
    
    # helper
    u_from_x = lambda x: x[:, 0, :16, :]
    u0_from_x = lambda x: x[:, 0, 0, :]
    f_from_x = lambda x: x[:, 1, :15, :]
    
    db = torch.load(f'/usr/{args.dataset}/test')
    target_step = db['u'][-50:].cuda()*0.1
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(0)
    ddpm_init = load_2dconv_model(model_i, args)
    
    total_time_steps = 80
    
    t1 = time.time()
    
    if args.asynch_inference_mode:
        assert args.infer_interval == 1
        args.is_init_model = False
        ddpm_feedback = load_2dconv_model(model_f, args)
                
        print('start initialization of online inference...')
        kwargs['u_final'] = target_step[:,0:15]
        x = ddpm_init.sample(**kwargs)
        kwargs['u_pred'] = anti_diagonal(x).permute(0,2,1,3)
        
        print("initialization of online inference finised.")
        for t in range(0, total_time_steps):  
            print('online step:', t)
            kwargs['u_final'] = target_step[:,t:t+15]
            x = ddpm_feedback.sample(**kwargs)*RESCALER  
            u_controlled = burgers_numeric_solve_free(kwargs['u_init']*RESCALER, f_from_x(x)[:,[0]], visc=0.01, T=0.1, dt=1e-4, num_t=1)
            ddpm_mse = mse_deviation(x[:,0,1], u_controlled[:,-1], partially_observed=args.partially_observed).cpu()            
            diffused_mse = (u_from_x(x)[:, 1, :] - kwargs['u_final'][:,0]*RESCALER).square().mean(-1) # MSE
            diffused_mse_median, _ = (u_from_x(x)[:, 1, :] - kwargs['u_final'][:,0]*RESCALER).square().median(-1)
            diffused_mae = (u_from_x(x)[:, 1, :] - kwargs['u_final'][:,0]*RESCALER).abs().mean(-1) # MAE
            diffused_mae_median, _ = (u_from_x(x)[:, 1, :] - kwargs['u_final'][:,0]*RESCALER).abs().median(-1)
            ep = 1e-5
            diffused_nmse = (u_from_x(x)[:, 1, :] - kwargs['u_final'][:,0]*RESCALER).square().mean(-1) / ((kwargs['u_final'][:,0]*RESCALER).square().mean() + ep) # normalized MSE
            diffused_nmae = (u_from_x(x)[:, 1, :] - kwargs['u_final'][:,0]*RESCALER).abs().mean(-1) / ((kwargs['u_final'][:,0]*RESCALER).abs().mean() + ep) # normalized MAE
            
            J_diffused = (diffused_mse, diffused_mse_median, diffused_mae, diffused_mae_median, diffused_nmse, diffused_nmae)
            
            mse = (u_controlled[:, -1, :] - kwargs['u_final'][:,0]*RESCALER).square().mean(-1) # MSE
            mse_median, _ = (u_controlled[:, -1, :] - kwargs['u_final'][:,0]*RESCALER).square().median(-1)
            mae = (u_controlled[:, -1, :] - kwargs['u_final'][:,0]*RESCALER).abs().mean(-1) # MAE
            mae_median, _ = (u_controlled[:, -1, :] - kwargs['u_final'][:,0]*RESCALER).abs().median(-1)
            ep = 1e-5
            nmse = (u_controlled[:, -1, :] - kwargs['u_final'][:,0]*RESCALER).square().mean(-1) / ((kwargs['u_final'][:,0]*RESCALER).square().mean() + ep) # normalized MSE
            nmae = (u_controlled[:, -1, :] - kwargs['u_final'][:,0]*RESCALER).abs().mean(-1) / ((kwargs['u_final'][:,0]*RESCALER).abs().mean() + ep) # normalized MAE
            
            J_actual = (mse, mse_median, mae, mae_median, nmse, nmae)
            energy = f_from_x(x)[:,[0]].square().sum((-1, -2))
            
            elems_to_cpu_numpy_if_tuple = lambda x: x.cpu().numpy() if type(x) is not tuple else np.array([xi.cpu().numpy() for xi in x])
            J_diffused = elems_to_cpu_numpy_if_tuple(J_diffused)
            J_actual = elems_to_cpu_numpy_if_tuple(J_actual)
            energy = energy.cpu().numpy()
            
            logs['states'].append(u_controlled)
            logs['ddpm_mse'].append(ddpm_mse)
            logs['J_diffused'].append(diffused_mse)
            logs['J_actual_mse'].append(mse)
            logs['J_actual_nmae'].append(nmae)
            logs['energy'].append(energy)
            logs['target'].append(kwargs['u_final'][:,0]*RESCALER)
            logs['f'].append(f_from_x(x)[:,[0]])
                            
            kwargs['u_init'] = u_controlled[:,-1]/RESCALER
            kwargs['u_pred'] = x[:,:,1:-1]/RESCALER
            
    else:  
        for t in range(0,total_time_steps):
            print('online step:', t)
    
            t_ = t%args.infer_interval
                
            if t_ == 0:
                print('make new noise')
                kwargs['u_final'] = target_step[:,t:t+15]
                x = ddpm_init.sample(**kwargs) * RESCALER
                        
            x_gt = burgers_numeric_solve_free(x[:,0,t_], f_from_x(x)[:,[t_]], visc=0.01, T=0.1, dt=1e-4, num_t=1)
            u_controlled = burgers_numeric_solve_free(kwargs['u_init']*RESCALER, f_from_x(x)[:,[t_]], visc=0.01, T=0.1, dt=1e-4, num_t=1)
            
            ddpm_mse = mse_deviation(x[:,0,t_+1], x_gt[:,-1], partially_observed=args.partially_observed).cpu()
            
            diffused_mse = (u_from_x(x)[:, t_+1, :] - kwargs['u_final'][:,t_]*RESCALER).square().mean(-1) # MSE
            diffused_mse_median, _ = (u_from_x(x)[:, t_+1, :] - kwargs['u_final'][:,t_]*RESCALER).square().median(-1)
            diffused_mae = (u_from_x(x)[:, t_+1, :] - kwargs['u_final'][:,t_]*RESCALER).abs().mean(-1) # MAE
            diffused_mae_median, _ = (u_from_x(x)[:, t_+1, :] - kwargs['u_final'][:,t_]*RESCALER).abs().median(-1)
            ep = 1e-5
            diffused_nmse = (u_from_x(x)[:, t_+1, :] - kwargs['u_final'][:,t_]*RESCALER).square().mean(-1) / ((kwargs['u_final'][:,t_]*RESCALER).square().mean() + ep) # normalized MSE
            diffused_nmae = (u_from_x(x)[:, t_+1, :] - kwargs['u_final'][:,t_]*RESCALER).abs().mean(-1) / ((kwargs['u_final'][:,t_]*RESCALER).abs().mean() + ep) # normalized MAE
            J_diffused = (diffused_mse, diffused_mse_median, diffused_mae, diffused_mae_median, diffused_nmse, diffused_nmae)
            
            mse = (u_controlled[:, -1, :] - kwargs['u_final'][:,t_]*RESCALER).square().mean(-1) # MSE
            mse_median, _ = (u_controlled[:, -1, :] - kwargs['u_final'][:,t_]*RESCALER).square().median(-1)
            mae = (u_controlled[:, -1, :] - kwargs['u_final'][:,t_]*RESCALER).abs().mean(-1) # MAE
            mae_median, _ = (u_controlled[:, -1, :] - kwargs['u_final'][:,t_]*RESCALER).abs().median(-1)
            ep = 1e-5
            nmse = (u_controlled[:, -1, :] - kwargs['u_final'][:,t_]*RESCALER).square().mean(-1) / ((kwargs['u_final'][:,t_]*RESCALER).square().mean() + ep) # normalized MSE
            nmae = (u_controlled[:, -1, :] - kwargs['u_final'][:,t_]*RESCALER).abs().mean(-1) / ((kwargs['u_final'][:,t_]*RESCALER).abs().mean() + ep) # normalized MAE
            
            J_actual = (mse, mse_median, mae, mae_median, nmse, nmae)
            energy = f_from_x(x)[:,[t_]].square().sum((-1, -2))
            
            elems_to_cpu_numpy_if_tuple = lambda x: x.cpu().numpy() if type(x) is not tuple else np.array([xi.cpu().numpy() for xi in x])
            J_diffused = elems_to_cpu_numpy_if_tuple(J_diffused)
            J_actual = elems_to_cpu_numpy_if_tuple(J_actual)
            energy = energy.cpu().numpy()
            
            logs['states'].append(u_controlled)
            logs['ddpm_mse'].append(ddpm_mse)
            logs['J_diffused'].append(diffused_mse)
            logs['J_actual_mse'].append(mse)
            logs['J_actual_nmae'].append(nmae)
            logs['energy'].append(energy)
            logs['target'].append(kwargs['u_final'][:,t_]*RESCALER)
            logs['f'].append(f_from_x(x)[:,[t_]])            

            kwargs['u_init'] = u_controlled[:,-1]/RESCALER
    
    t2 = time.time()
    
    print('****************************************************************************')
    print('Total time 0f 50 batch size in second:', t2 - t1)
    print('#############################################################################')
    
    return ddpm_mse, J_diffused, J_actual, energy


def get_scheduler(scheduler):
    if scheduler is None:
        return None
    # decreasing schedules
    elif scheduler == 'linear':
        raise NotImplementedError
    elif scheduler == 'cosine':
        return cosine_beta_J_schedule
    elif scheduler == 'plain_cosine':
        return plain_cosine_schedule
    elif scheduler == 'sigmoid':
        return sigmoid_schedule
    # increasing step (eta[t=0] is the largest)
    elif scheduler == 'sigmoid_flip':
        return sigmoid_schedule_flip
    else:
        raise ValueError
    

def evaluate(
        model_i, 
        model_f,
        args, 
        rep=1, 
        wu=0, 
        wf=0, 
        wpinn=0, 
        wf_eval=0, 
        wu_eval=1, 
        conv2d=True, 
):
    batch_size = 50 // rep
    mses = []
    l_gts = []
    l_dfs = []
    energies = []
    for i in range(rep):
        seed = i
        target_idx = list(range(i * batch_size, (i + 1) * batch_size)) # should also work if being an iterable
        if conv2d:
            ddpm_mse, J_diffused, J_actual, energy = diffuse_2dconv(
                args, 
                custom_metric=lambda f, **kwargs: metric(
                    get_target(target_idx, dataset=args.dataset), 
                    f, 
                    target='final_u', 
                    partial_control=args.partial_control, 
                    report_all=True, 
                    partially_observed=args.partially_observed, 
                    **kwargs
                ), 
                model_i=model_i, 
                model_f = model_f,
                seed=seed, 
                # more ddpm settings
                nablaJ=get_nablaJ_2dconv(
                    target_i=target_idx, 
                    wu=wu, wf=wf, wpinn=wpinn, 
                    wf_eval=wf_eval, wu_eval=wu_eval, 
                    dist_reg=mse_dist_reg, wreg=args.wreg,
                    dataset=args.dataset, 
                    partially_observed=args.partially_observed, 
                    pinn_loss_mode=args.pinn_loss_mode, 
                ),  
                J_scheduler=get_scheduler(args.J_scheduler), 
                w_scheduler=get_scheduler(args.w_scheduler), 
                # proj_guidance=get_proj_ep_orthogonal_func(norm='F'), 
                clip_denoised=True, 
                guidance_u0=True, 
                batch_size=batch_size,
                u_init=get_target(
                    target_idx, 
                    dataset=args.dataset, 
                    partially_observed_fill_zero_unobserved=args.partially_observed
                )[:, 0, :] / RESCALER, 
                u_final=None,
                u_pred = None,
            )
        else:
            raise NotImplementedError('2D conv seems to be the best for now')
        mses.append(ddpm_mse)
        l_gts.append(J_actual)
        l_dfs.append(J_diffused)
        energies.append(energy)

    return np.stack(mses).mean(0), *(np.stack(l_dfs).mean(0)[i] for i in range(l_dfs[0].shape[0])), *(np.stack(l_gts).mean(0)[i] for i in range(l_dfs[0].shape[0])), np.stack(energies).mean(0)
    
def save_eval_results(*save_values, model_i, fname, model_str='', partially_observed=None):
    names = [
        'mse_gt', 
        'J_diffused_mse', 'J_diffused_mse_median', 'J_diffused_mae', 'J_diffused_mae_median', 'J_diffused_nmse', 'J_diffused_nmae', 
        'J_actual_mse', 'J_actual_mse_median', 'J_actual_mae', 'J_actual_mae_median', 'J_actual_nmse', 'J_actual_nmae', 
        'energy'
    ]

    for acc, inner_str in zip(save_values, names):
        save_acc(
            acc, 
            fname, 
            make_dict_path=lambda acc, dict_args: {
                dict_args['model_name']: {
                    'model_description': dict_args['model_str'], 
                    dict_args['guidance_str']: {
                        inner_str: acc
                    }
                }
            }, 
            model_name=f'{model_i}', 
            model_str=model_str, 
            guidance_str=f'wu={wu:.1f}, wf={wf:.1f}, wpinn={wpinn:.1f}' # these values are from the outer scope
        )

if __name__ == '__main__':
    args = parser.parse_args()

    model_i, model_f, model_str, conv_2d = args.exp_id_i, args.exp_id_f, args.model_str, True
    
    logs = dict()
    logs['states'] = []
    logs['target'] = []
    logs['ddpm_mse'] = []
    logs['J_diffused'] = []
    logs['J_actual_mse'] = []
    logs['J_actual_nmae'] = []
    logs['J_relative'] = []
    logs['energy'] = []  
    logs['f'] = []  

    for wf in args.wfs: 
        for wu in args.wus:
            for wpinn in args.wpinns:
                results = evaluate(
                    model_i = model_i, 
                    model_f = model_f,
                    args = args,
                    rep = 1, 
                    wu = wu, 
                    wf = wf, 
                    wpinn = wpinn, 
                    conv2d = conv_2d, 
                )
                save_eval_results(
                    *results, 
                    model_i = model_i + args.model_str_in_key, 
                    fname = args.save_file, 
                    model_str = model_str + ' u0 guidance, rescaler 10', 
                    partially_observed=args.partially_observed, 
                )