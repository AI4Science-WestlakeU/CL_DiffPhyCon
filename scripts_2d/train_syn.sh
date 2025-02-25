data_path="/data/cl_diffphycon/2d"
accelerate launch --config_file default_config.yaml \
--main_process_port 29500 \
--gpu_ids 0,1 \
../train/train_2d.py \
--results_path "${data_path}/checkpoints/syn_models" \
--dataset_path ${data_path} \
--is_synch_model