# Cell 3: Stage 2 DWI consistency-model testing
# UPDATE data_root TO YOUR DATA PATH e.g. ".../data/HARDI150.nii.gz"
# UPDATE valid_mask TO YOUR VALID MASK START AND END, default is 10 160
# UPDATE resume_checkpoint TO THE STAGE II CHECKPOINT PATH e.g. ".../save/stageII/model300000.pt"
# UPDATE noisemodel_checkpoint TO THE STAGE I CHECKPOINT PATH e.g. ".../save/stageI/model150000.pt"
# UPDATE save_dir TO YOUR SAVING PATH FOR INFERENCE e.g. ".../save/inference"

python -m scripts.cm_test_stage2_dwi \
  --data_root null \
  --valid_mask 10 160 \
  --resume_checkpoint null \
  --noisemodel_checkpoint null \
  --save_dir null \
  --image_size 128 \
  --in_channels 1 \
  --out_channels 1 \
  --num_channels 32 \
  --channel_mult 1,2,4,8,8 \
  --num_res_blocks 2 \
  --attention_resolutions 16 \
  --num_heads 4 \
  --num_head_channels 32 \
  --num_heads_upsample 4 \
  --dropout 0.0 \
  --use_scale_shift_norm False \
  --resblock_updown True \
  --use_fp16 True \
  --use_new_attention_order False \
  --learn_sigma False \
  --sigma_min 0.002 \
  --sigma_max 10 \
  --weight_schedule uniform \
  --no-save_pngs \
  --save_nifti \
  --device cuda
