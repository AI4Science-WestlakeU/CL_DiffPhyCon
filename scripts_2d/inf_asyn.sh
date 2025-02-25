data_path="/data/cl_diffphycon/2d"
python ../inference/inference_2d.py \
--dataset_path "${data_path}" \
--inference_result_path "${data_path}/inference_results/" \
--init_diffusion_model_path "${data_path}/checkpoints/unet_syn_bsize12_cond_d_v_s_diffsteps600_horizon15" \
--online_diffusion_model_path "${data_path}/checkpoints/unet_asyn_bsize12_cond_d_v_s_diffsteps600_horizon15/" \
--using_ddim True \
--asynch_inference_mode
