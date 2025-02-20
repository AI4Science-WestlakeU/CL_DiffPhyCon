exp_id_ls=(0)

for i in "${!exp_id_ls[@]}"; do
    exp_id=${exp_id_ls[i]}
    dim=${dim_ls[i]}
    CUDA_VISIBLE_DEVICES=0 python ../inference/inference_1d.py \
    --exp_id_i time_vary_twomodel_syn \
    --exp_id_f time_vary_twomodel_asyn \
    --dataset free_u_f_1e5_seed0 \
    --is_condition_u0 True \
    --is_condition_uT True \
    --save_file burgers_results/full_obs_full_ctr/uT_condition/test.yaml \
    --infer_interval 1 \
    --checkpoint 9 \
    --eval_save time_vary_asyn1_5 \
    --asynch_inference_mode 
done
