# run stage1
# Cell 1: Stage 1 DWI denoiser training
# UPDATE data_root TO YOUR DATA PATH e.g. ".../data/HARDI150.nii.gz"
# UPDATE save_dir TO YOUR SAVING PATH e.g. ".../save/stageI"

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
  --epoch 157 \
  --log_interval 500 \
  --channel_mult 1,2,4,8,8
