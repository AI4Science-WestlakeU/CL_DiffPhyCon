import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from diffusion.diffusion_2d import GaussianDiffusion, Trainer
from dataset.data_2d import Smoke
import pdb
import torch
from accelerate import Accelerator
import datetime

import argparse

from IPython import embed

def load_model_accelerator(model, model_path):
    fp16 = False
    accelerator = Accelerator(
        split_batches = True,
        mixed_precision = 'fp16' if fp16 else 'no'
    )
    device = accelerator.device
    data = torch.load(model_path, map_location=device)
    model = accelerator.unwrap_model(model)
    model.load_state_dict(data)
    
    return model


parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--dataset', default='Smoke', type=str,
                    help='dataset to evaluate')
parser.add_argument('--dataset_path', default="/data", type=str,
                    help='path to dataset')
parser.add_argument('--is_condition_control', default=False, type=eval,
                    help='If condition on control')
parser.add_argument('--is_condition_pad', default=True, type=eval,
                    help='If condition on padded state')
parser.add_argument('--batch_size', default=12, type=int,
                    help='size of batch of input to use')
parser.add_argument('--horizon', default=15, type=int,
                    help='number of horizon to diffuse')
parser.add_argument('--train_num_steps', default=250000, type=int,
                    help='total training steps')
parser.add_argument('--results_path', default="./results/train/{}/".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), type=str,
                    help='folder to save training checkpoints')
parser.add_argument('--diffusion_steps', default=600, type=int,
                    help='number of denoising steps in diffusion model')
parser.add_argument('--is_synch_model', action='store_true', help="whether use synchronous denoising steps among different time steps")


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    print(FLAGS)
    
    # get shape, RESCALER
    if FLAGS.dataset == "Smoke":
        dataset = Smoke(
            dataset_path=FLAGS.dataset_path,
            horizon=FLAGS.horizon,
            is_train=True,
        )
        _, shape, ori_shape, _ = dataset[0]
    else:
        assert False
    RESCALER = dataset.RESCALER.unsqueeze(0)

    from model.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D
    model = Unet3D_with_Conv3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels = 6,
    )

    print("Number of parameters: {}".
          format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("Saved at: ", FLAGS.results_path)

    diffusion = GaussianDiffusion(
        model,
        RESCALER,
        FLAGS.is_condition_control,
        FLAGS.is_condition_pad,
        ori_shape,
        image_size = 64,
        horizon = FLAGS.horizon, 
        diffusion_steps = FLAGS.diffusion_steps,           # number of diffusion steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l2',            # L1 or L2
        objective = "pred_noise",
        is_synch_model = FLAGS.is_synch_model
    )

    trainer = Trainer(
        diffusion,
        FLAGS.dataset,
        FLAGS.dataset_path,
        train_batch_size = FLAGS.batch_size,
        train_lr = 1e-3, 
        train_num_steps = FLAGS.train_num_steps, # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = 10000, # 10000
        horizon = FLAGS.horizon,
        results_path = FLAGS.results_path,
        amp = False,                       # turn on mixed precision
        calculate_fid = False,              # whether to calculate fid during training
    )

    trainer.train()