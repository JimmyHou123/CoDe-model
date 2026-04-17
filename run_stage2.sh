#!/bin/bash
# Cell 2: Stage 2 DWI consistency-model training
# UPDATE data_root TO YOUR DATA PATH e.g. ".../data/HARDI150.nii.gz"
# If you do not intend to use the RMT regularizer, set the loss_norm option to l1, l2, or lpips
# UPDATE save_dir TO YOUR SAVING PATH e.g. ".../save/stageII"
# UPDATE noisemodel_checkpoint TO THE STAGE I CHECKPOINT PATH e.g. ".../save/stageI/model150000.pt"
# UPDATE valid_mask_start TO YOUR VALID MASK START, default is 10
# UPDATE valid_mask_end TO YOUR VALID MASK END, default is 160
# UPDATE rmt_npy_path TO THE RMT DENOISED ARRAY e.g. ".../rmt_denoised.npy"

mpiexec -n 1 --allow-run-as-root python -m scripts.cm_train_stage2_dwi \
  --data_root null \
  --training_mode consistency_training \
  --target_ema_mode adaptive \
  --start_ema 0.95 \
  --scale_mode progressive \
  --start_scales 2 \
  --end_scales 150 \
  --total_training_steps 300000 \
  --loss_norm lpips_rmt \
  --lr_anneal_steps 0 \
  --attention_resolutions 16 \
  --use_scale_shift_norm False \
  --dropout 0.0 \
  --teacher_dropout 0.1 \
  --ema_rate 0.9999,0.99994,0.9999432189950708 \
  --global_batch_size 2 \
  --image_size 128 \
  --lr 0.00005 \
  --noisy_in_channels 2 \
  --in_channels 1 \
  --out_channels 1 \
  --num_channels 32 \
  --num_head_channels 32 \
  --num_res_blocks 2 \
  --resblock_updown True \
  --schedule_sampler uniform \
  --use_fp16 True \
  --weight_decay 0.0 \
  --weight_schedule uniform \
  --save_dir null \
  --save_interval 10000 \
  --epoch 53 \
  --channel_mult 1,2,4,8,8 \
  --sigma_min 0.0002 \
  --sigma_max 2 \
  --noisemodel_checkpoint null \
  --log_interval 500 \
  --valid_mask_start 10 \
  --valid_mask_end 160 \
  --rmt_npy_path null
