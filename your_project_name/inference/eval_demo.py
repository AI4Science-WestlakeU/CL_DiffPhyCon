#!/usr/bin/env python
# coding: utf-8

# # Load packages
# 

# In[ ]:


try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass
# general package

import matplotlib.pyplot as plt
import os
import sys
import argparse
import tqdm
from einops import rearrange, repeat
import logging
import datetime  
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets 
import numpy as np
#custom package, model,data,utils
from standard_repo.data.data_demo import Advection
from standard_repo.model.model_demo import Net_demo
from standard_repo.utils.utils import set_seed,caculate_confidence_interval
# path
from standard_repo.filepath import EXP_PATH,SRC_PATH,PARENT_WP


# # Set Arguments
# 

# In[ ]:


parser = argparse.ArgumentParser(description="Training Configurations of autoencoder fot theory")
parser.add_argument("--date_exp", type=str, default="2021-09-30", help="Date of the experiment")
parser.add_argument("--exp_name", type=str, default="AE_overfit", help="Name of the experiment")
parser.add_argument("--dataset_path", type=str, default="data", help="Path to the data")
parser.add_argument("--results_path", type=str, default="results", help="Path to the results")
parser.add_argument('--config', type=str, default='/project_module/configs/config.yaml')


# evaluation configurations
parser.add_argument("--eval_batch_size", type=int, default=256, help="eval Batch size")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path load the checkpoint to restore, if not None, contine training")


parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU")
parser.add_argument("--seed", type=int, default=0, help="Seed for the random number generator")

# model configurations



args = parser.parse_args([])

args.date_exp="2024-08-05"
args.exp_name="eval_demo_test"
args.config = "standard_repo/results/2024-08-05/taining_demo/config.yaml"

args.dataset_path ="standard_repo/dataset/advection"
# training configurations
args.eval_batch_size = 512

# configure environment
args.gpu_id = 0
args.seed = 42


args.config = os.path.join(PARENT_WP,args.config)
if args.config!=None:
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)

    # 更新 args
    for key, value in config_data.items():
        try:
            setattr(args, key, value)
        except:
            pass
args.checkpoint_path = "standard_repo/results/2024-08-05/taining_demo/model_epoch_20.pth"
args.checkpoint_path = os.path.join(PARENT_WP,args.checkpoint_path)   
# set up logging
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = os.path.join(args.results_path, "evaluation_{}.log".format(current_time))
logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('args: {}'.format(args))
# set device and seed
device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
set_seed(args.seed)


# # Load data and Cingure model

# In[ ]:


# load dataset
eval_dataset = Advection(
        dataset_name="Advection",
        dataset_path=args.dataset_path,
        mode = 'test',
        input_steps=1,
        output_steps=80,
        time_interval=1,
        simutime_steps=80,
        rescaler=4,
    )
eval_dataloader = torch.utils.data.DataLoader(eval_dataset,batch_size= args.eval_batch_size, shuffle=False, pin_memory=True,num_workers=8)
logging.info(f"data loaded from{args.dataset_path}")

# configure model
model = Net_demo().to(device)
if args.checkpoint_path is not None:
    model.load_state_dict(torch.load(args.checkpoint_path))
    logging.info(f"Checkpoint{args.checkpoint_path} loaded")

criterion = nn.MSELoss()


# # Evaluation

# In[ ]:


logging.info("Start evaluate on ",torch.cuda.get_device_name())
num_params = sum(p.numel() for p in model.parameters())
logging.info("Number of parameters: {}".format(num_params))
start_time = time.time()
bs= args.eval_batch_size
with torch.no_grad():
    for eval_data in eval_dataloader:
        break
    input,target = eval_data
    input = input.to(device)
    target = target.to(device)
    output = model(input) # [B,1,T,s] => [B,1,T,s]
end_time = time.time()
relative_mse =torch.norm(target- output) / torch.norm(target)
target = target.reshape(bs,-1)
output = output.reshape(bs,-1)
mse_bach = ((target-output)**2).mean(-1) #[B,]
mean,std_dev,margin_of_error,min_value=caculate_confidence_interval(mse_bach)
logging.info("Evaluate the checkpoint: {}".format(args.checkpoint_path))
logging.info("Relative MSE: {:.8f}".format(relative_mse))
logging.info("MSE: {:.8f}".format(mean))
logging.info("MSE std: {:.8f}".format(std_dev))
logging.info("MSE margin of error: {:.8f}".format(margin_of_error))
logging.info(f"Evaulation complete, time cost is {end_time-start_time}s")
logging.info("Results save at {}".format(args.results_path))

