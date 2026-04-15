# UPDATE data_root TO YOUR DATA PATH e.g. ".../data/HARDI150.nii.gz"
# UPDATE save_dir TO YOUR SAVING PATH e.g. ".../save/RMT_denoised"
# UPDATE valid_mask_start TO YOUR VALID MASK START, default is 10
# UPDATE valid_mask_end TO YOUR VALID MASK END, default is 160

python dipy_mppca.py \
  --data_root null \
  --save_dir null \
  --valid_mask_start 10 \
  --valid_mask_end 160 \
  --patch_radius 3
