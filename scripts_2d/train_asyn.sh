data_path="/data/closed_loop_diffcon/phiflow"
train_results_folder="$asyn_cond_d_v_s"
accelerate launch --config_file default_config.yaml \
--main_process_port 29501 \
--gpu_ids 4,5 \
../train/train_2d.py \
--results_path "${data_path}/checkpoints/${train_results_folder}" \
--dataset_path ${data_path} 