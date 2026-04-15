python dipy_mppca.py \
  # UPDATE THIS TO YOUR DATA PATH e.g. ".../data/HARDI150.nii.gz"
  --data_root null \
  # UPDATE THIS TO YOUR SAVING PATH e.g. ".../save/RMT_denoised"
  --save_dir null \
  # UPDATE THIS TO YOUR VALID MASK START, default is 10
  --valid_mask_start 10 \
  # UPDATE THIS TO YOUR VALID MASK END, default is 160
  --valid_mask_end 160 \
  --patch_radius 3
