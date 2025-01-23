import torch
import numpy as np
import yaml
import argparse
from datetime import datetime
import os
import sys
from termcolor import colored
import pdb
import matplotlib.pyplot as plt
COLOR_LIST = ["b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine",
             "b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine"]

#basic utils
class Printer(object):
    ## Example, to print code running time between two p.print() calls
    # p.print(f"test_start", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """
        
        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(self, item, tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=-1, precision="second", is_silent=False):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2], avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window+1,len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, 'yellow'))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))
def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time
    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string
p = Printer(n_digits=6)

# config utils
def add_args_from_config(parser):
    args=parser.parse_args()
    with open(args.config, 'r') as file:
        config=yaml.safe_load(file)

    existing_args = {action.dest for action in parser._actions}
    
    # Add arguments from the config file to the parser
    for key, value in config.items():
        # If the argument is not already added, add it to the parser
        if key not in existing_args:
            parser.add_argument(f'--{key}', type=type(value), default=value)
    
    return parser
def save_config_from_args(args):
    config_dict = {k: v for k, v in vars(args).items() if k != 'config'}
    
    config_dir = args.results_path
    os.makedirs(config_dir, exist_ok=True)  # Ensure the directory exists
    config_file_path = os.path.join(config_dir, 'config.yaml')
    with open(config_file_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)

    return 


# result analysis
def caculate_confidence_interval(data):
    ''''
    input example: abs(pred_design-pred_simu)
    '''
    list_dim=range(data.dim())
    if data.dim()>1:
        MAE_batch_size = torch.mean(data, dim=tuple(list_dim[1:]))
    else:
        MAE_batch_size =data
    mean = torch.mean(MAE_batch_size)
    
    std_dev = torch.std(MAE_batch_size)
    min_value=min(MAE_batch_size)
    confidence_level = 0.95
    # pdb.set_trace()
    n = len(data)
    # kk = stats.t.ppf((1 + confidence_level) / 2, n - 1) * (std_dev / (n ** 0.5))
    margin_of_error= std_dev* 1.96/ torch.sqrt(torch.tensor(n,dtype=float))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    print("mean:", mean.item())
    print("std:", std_dev.item())
    print(f"margin_of_error:", margin_of_error)

    return mean,std_dev,margin_of_error,min_value


#training utils
def caculate_num_parameters(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    # pdb.set_trace()
    return
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def draw_loss(training_loss_list, test_loss_list, save_path):
    """
    Plot the training and testing loss curves and save the plot to a file.
    
    Args:
    training_loss_list (list): List of training losses.
    test_loss_list (list): List of testing losses.
    save_path (str): Path where the plot will be saved.
    
    Returns:
    None
    """
    # Create a figure and axis object using plt.subplots
    fig, ax = plt.subplots()

    # Plot training and testing loss
    ax.plot(training_loss_list, label='Training Loss')
    ax.plot(test_loss_list, label='Testing Loss')

    # Set title and labels
    ax.set_title('Training and Testing Loss During Training')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

    # Display legend
    ax.legend()

    # Save the plot to the specified file
    plt.savefig(save_path)

    # Show the plot
    plt.show()


def get_entropy(X, Y=None, K=3, NN=100, normalize=False, stop_grad_reference=False):
    '''
    Estimating differential entropy with K-nearest neighbors.
    From Lombardi, Damiano, and Sanjay Pant. "Nonparametric k-nearest-neighbor entropy estimator."
        Physical Review E 93.1 (2016): 013310.

    Args:
        X, Y: torch.tensor, with shape of [N, F] or [N], where N is the number of samples,
            and F is the feature size. If Y is not None, then compute the entropy of H(X, Y).
        K: K^th nearest neighbor. Slightly larger K yields more stable estimates.
        NN: number of samples to sample from to compute the entropy.
        normalize: whether to normalize the X (and Y). Default False.
        stop_grad_reference: if True, will stop gradient for the reference samples.

    Returns:
        entropy: scalar, estimated entropy.
    '''
    epsilon = 1e-20
    MAX = 1e10

    if len(X.shape) > 2:
        raise Exception('The shape of X and Y must be [N, F] or [N]. Please fix the shape!')
    device = X.device
    NN = min(NN, X.shape[0]) # number of points to sample from
    K = min(K, NN)
    indices = torch.randperm(X.shape[0])[:NN]
    if X.ndim == 1:
        X = X[indices, None]
    else:
        X = X[indices]
    if normalize:
        X = (X - X.mean(0)) / (X.std(0) + epsilon)

    if Y is not None:
        if Y.ndim == 1:
            Y = Y[indices, None]
        else:
            Y = Y[indices]
        if normalize:
            Y = (Y - Y.mean(0)) / (Y.std(0) + epsilon)
        X = torch.cat((X, Y), dim=1)
    if stop_grad_reference:
        dist_matrix = torch.norm(X[:, None, :] - X[None, :, :].detach(), p=2, dim=-1)  # dist_matrix: [N, N]
    else:
        dist_matrix = torch.norm(X[:, None, :] - X[None, :, :], p=2, dim=-1)  # dist_matrix: [N, N]
    dist_matrix = torch.where(
        torch.eye(dist_matrix.size(0), dtype=bool, device=device),
        torch.tensor(MAX, dtype=dist_matrix.dtype, device=device),
        dist_matrix
    )  # Assign the diagonal elements to be MAX.
    top_k_values = torch.topk(dist_matrix, K, dim=1, largest=False)[0][:, -1] # [N], obtaining the top K'th distance from each row
    return X.shape[-1] * torch.log(top_k_values + epsilon).mean(0) + torch.digamma(torch.tensor(NN, device=device)) \
            - torch.digamma(torch.tensor(K, device=device)) + torch.log(torch.tensor(torch.pi, device=device)) * X.shape[-1] / 2 \
            - torch.lgamma(torch.tensor(1 + X.shape[-1] / 2, device=device))


def get_mi(X, Y, K=3, NN=100, normalize=False, stop_grad_reference=False):
    '''
    Compute estimation for mutual information.

    Args:
        X, Y: torch.tensor, with shape of [N, F] or [N], where N is the number of samples,
            and F is the feature size. If Y is not None, then compute the entropy of H(X, Y).
        K: K^th nearest neighbor. Slightly larger K yields more stable estimates.
        NN: number of samples to sample from to compute the entropy.
        normalize: whether to normalize the X (and Y). Default False.
        stop_grad_reference: if True, will stop gradient for the reference samples.

    Returns:
        mi: scalar, estimated mutual information.
    '''
    ent_X = entropy(X, K=K, NN=NN, normalize=normalize, stop_grad_reference=stop_grad_reference)
    ent_Y = entropy(Y, K=K, NN=NN, normalize=normalize, stop_grad_reference=stop_grad_reference)
    ent_XY = entropy(X, Y, K=K, NN=NN, normalize=normalize, stop_grad_reference=stop_grad_reference)
    # print(ent_X, ent_Y, ent_XY)
    mutual_information = ent_X + ent_Y - ent_XY
    return mutual_information


def get_hashing(string_repr, length=None):
    """Get the hashing of a string."""
    import hashlib, base64
    hashing = base64.b64encode(hashlib.md5(string_repr.encode('utf-8')).digest()).decode().replace("/", "a")[:-2]
    if length is not None:
        hashing = hashing[:length]
    return hashing
