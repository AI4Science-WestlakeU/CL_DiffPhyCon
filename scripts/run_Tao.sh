#activate the conda environment
source /opt/conda/bin/activate
conda activate ideEnv

# Run the python script
python standard_repo/train/train_demo.py --dataset_path "dataset/advection" --date_exp '2024-09-08'
