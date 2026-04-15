# UPDATE THIS TO YOUR DATA PATH e.g. ".../data/HARDI150.nii.gz"
# UPDATE THIS TO YOUR SAVING PATH e.g. ".../save/RMT_denoised"
# UPDATE THIS TO YOUR VALID MASK START, default is 10
# UPDATE THIS TO YOUR VALID MASK END, default is 160

python dipy_mppca.py \
  --data_root null \
  --save_dir null \
  --valid_mask_start 10 \
  --valid_mask_end 160 \
  --patch_radius 3
