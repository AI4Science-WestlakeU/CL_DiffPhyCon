import argparse
import yaml
from utils.utils import add_args_from_config, save_config_from_args, p
import pdb
from filepath import EXP_PATH, CURRENT_WP

if __name__ == "__main__":
    # Create the parser from the YAML config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser = add_args_from_config(parser)
    # Parse the arguments
    args = parser.parse_args()
    args.exp_path = EXP_PATH
    # %%
    p.print(
        f"test_start",
        tabs=0,
        is_datetime=None,
        banner_size=0,
        end=None,
        avg_window=1,
        precision="millisecond",
        is_silent=False,
    )
    save_config_from_args(args)
    # Print the parsed arguments
    print(args)
