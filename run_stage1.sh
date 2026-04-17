#!/bin/bash
# run stage1
# Cell 1: Stage 1 DWI denoiser training
# UPDATE data_root TO YOUR DATA PATH e.g. ".../data/HARDI150.nii.gz"
# UPDATE save_dir TO YOUR SAVING PATH e.g. ".../save/stageI"
# UPDATE valid_mask_start TO YOUR VALID MASK START, default is 10
# UPDATE valid_mask_end TO YOUR VALID MASK END, default is 160

mpiexec -n 1 --allow-run-as-root python -m scripts.cm_train_stage1_dwi \
  --data_root null \
  --total_training_steps 150000 \
  --lr_anneal_steps 0 \
  --attention_resolutions 16 \
  --use_scale_shift_norm False \
  --dropout 0.0 \
  --global_batch_size 4 \
  --image_size 128 \
  --lr 0.000005 \
  --in_channels 2 \
  --out_channels 1 \
  --num_channels 32 \
  --num_head_channels 32 \
  --num_res_blocks 2 \
  --resblock_updown True \
  --use_fp16 True \
  --weight_decay 0.0 \
  --save_dir null \
  --save_interval 10000 \
  --epoch 53 \
  --log_interval 500 \
  --channel_mult 1,2,4,8,8 \
  --valid_mask_start null \
  --valid_mask_end null
