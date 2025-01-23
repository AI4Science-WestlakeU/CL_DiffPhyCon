#!/usr/bin/env python
# coding: utf-8

# # Import package

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
from torch.utils.tensorboard import SummaryWriter
import datasets 
import numpy as np
#custom package
from standard_repo.data.data_demo import Advection
from standard_repo.model.model_demo import Net_demo
from standard_repo.utils.utils import set_seed,draw_loss,add_args_from_config,save_config_from_args
# path
from standard_repo.filepath import EXP_PATH,SRC_PATH,PARENT_WP


# # Augument

# In[ ]:


parser = argparse.ArgumentParser(description="Training Configurations of autoencoder fot theory")
parser.add_argument("--date_exp", type=str, default="2021-09-30", help="Date of the experiment")
parser.add_argument("--exp_name", type=str, default="AE_overfit", help="Name of the experiment")
parser.add_argument("--dataset_path", type=str, default="data", help="Path to the data")
parser.add_argument("--results_path", type=str, default="results", help="Path to the results")
parser.add_argument('--config', type=str, default=None, help='Path to the config file')


# training configurations
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--save_every", type=int, default=20, help="Save the model every x epochs")
parser.add_argument("--train_batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--test_batch_size", type=int, default=256, help="test Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path load the checkpoint to restore, if not None, contine training")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for the dataloader")

parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU")
parser.add_argument("--seed", type=int, default=0, help="Seed for the random number generator")
parser.add_argument("--is_use_tfb", type=bool, default=True, help="Whether to use tensorboard")
# model configurations


current_path = os.getcwd()
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args([])

    args.date_exp="2024-09-08"
    args.exp_name="taining_demo_test"
    args.config = "standard_repo/standard_repo/configs/config.yaml"

    args.dataset_path ="standard_repo/dataset/advection"
    # training configurations
    args.epochs = 10
    args.save_every = 1
    args.train_batch_size = 512
    args.test_batch_size = 512
    args.lr = 0.001
    args.checkpoint_path = None
    args.num_workers = 0

    # configure environment
    args.gpu_id = 0
    args.seed = 42


except:
    # parser = add_args_from_config(parser)
    args=parser.parse_args()
    if args.config!=None:
        with open(args.config, 'r') as file:
            config_data = yaml.safe_load(file)

        # 更新 args
        for key, value in config_data.items():
            setattr(args, key, value)
    is_jupyter = False

# prepare t path
args.results_path=EXP_PATH+"/results/"+args.date_exp+"/"+args.exp_name+"/"
args.dataset_path = os.path.join(EXP_PATH,args.dataset_path)
if args.config != None:
    args.config = os.path.join(EXP_PATH,args.config)

if args.checkpoint_path != None:
    args.checkpoint_path = os.path.join(EXP_PATH,args.checkpoint_path)
if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)
save_config_from_args(args)
# set up logging
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = os.path.join(args.results_path, "training_{}.log".format(current_time))
if args.is_use_tfb:
    writer = SummaryWriter(log_dir=args.results_path, filename_suffix="training_tf{}".format(current_time))
logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('args: {}'.format(args))
# set device and seed
device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
set_seed(args.seed)



# # Basic func

# In[ ]:


def cycle(dl):
    while True:
        for data in dl:
            yield data
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# # Load data and initialize model, model optimizer, loss function

# In[ ]:


# load dataset
train_dataset = Advection(
        dataset_name="Advection",
        dataset_path=args.dataset_path,
        mode = 'train',

        input_steps=1,
        output_steps=80,
        time_interval=1,
        simutime_steps=80,
        rescaler=4,
    )
test_dataset = Advection(
        dataset_name="Advection",
        dataset_path=args.dataset_path,
        mode = 'test',
        input_steps=1,
        output_steps=80,
        time_interval=1,
        simutime_steps=80,
        rescaler=4,
    )
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size= args.train_batch_size, shuffle=True, pin_memory=True,num_workers=args.num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True,num_workers=args.num_workers)
logging.info(f"data loaded from{args.dataset_path}")

# configure model
model = Net_demo().to(device)
if args.checkpoint_path is not None:
    model.load_state_dict(torch.load(args.checkpoint_path))
    logging.info(f"Checkpoint{args.checkpoint_path} loaded")

# configue optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.MSELoss()


# # Training model

# In[ ]:


logging.info("Start training on ",torch.cuda.get_device_name())
num_params = sum(p.numel() for p in model.parameters())
logging.info("Number of parameters: {}".format(num_params))
logging.info("number of batch in train_loader: ", len(train_dataloader))
print("number of batch in test_loader: ", len(test_dataloader))
start_time = time.time()
training_loss_list = []
test_loss_list = []
best_epoch = 0

# training loop
for epoch in tqdm.tqdm(range(1,args.epochs+1)):
    model.train()
    total_loss = 0.
    best_test_loss = 1e9
    for batch_idx, data in enumerate(train_dataloader):
        input,target = data
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input) # [B,1,T,s] => [B,1,T,s]
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.sum().item()
        # input = input.cpu()
        # target = target.cpu()
        torch.cuda.empty_cache()
    scheduler.step()
    
    average_loss = total_loss/len(train_dataloader)
    training_loss_list.append(average_loss)
    logging.info("training epoch {}, average loss: {}".format(epoch, average_loss))
    with torch.no_grad():
        for test_data in test_dataloader:
            break
        input,target = test_data
        input = input.to(device)
        target = target.to(device)
        output = model(input) # [B,1,T,s] => [B,1,T,s]
        loss = criterion(output, target)
        test_loss = loss.sum().item()
        test_loss_list.append(test_loss)
        if test_loss< best_test_loss:
            best_epoch = epoch
            best_test_loss = test_loss
        logging.info("testing epoch {}, loss: {}".format(epoch, test_loss))
    if args.is_use_tfb:
        writer.add_scalar("Loss/train", average_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
    if epoch % args.save_every == 0:
        torch.save(model.state_dict(), os.path.join(args.results_path, "model_epoch_{}.pth".format(epoch)))
        draw_loss(training_loss_list,test_loss_list,args.results_path)
end_time = time.time()
logging.info(f"Training complete, best epoch is {best_epoch}, time cost is {end_time-start_time}s")
logging.info("Results save at {}".format(args.results_path))
if args.is_use_tfb:
    writer.close()

# # Evaluate the model
