exp_id_ls=(0)

for i in "${!exp_id_ls[@]}"; do
    exp_id=${exp_id_ls[i]}
    dim=${dim_ls[i]}
    CUDA_VISIBLE_DEVICES=0 python ../inference/inference_1d.py \
    --exp_id_i model_syn \
    --exp_id_f model_asyn \
    --dataset '/usr/test_data' \
    --test_target '/usr/test_data' \
    --is_condition_u0 True \
    --is_condition_uT True \
    --save_file /usr/inference_savepath/test.yaml \
    --infer_interval 1 \
    --checkpoint 9 \
    --eval_save /usr/inference_savepath \
    --diffusion_model_path '/usr/checkpoints' \
    --asynch_inference_mode 
done
