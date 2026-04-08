import argparse
import os
import numpy as np

from dipy.denoise.localpca import mppca
from cm.dwi_datasets import MRIDataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    parser.add_argument('--valid_mask_start', type=int, default=10)
    parser.add_argument('--valid_mask_end', type=int, default=160)
    parser.add_argument('--patch_radius', type=int, default=3)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    ds = MRIDataset(
        dataroot=args.data_root,
        valid_mask=[args.valid_mask_start, args.valid_mask_end],
        phase='val',
        image_size=128,
        in_channel=1,
        val_volume_idx=40,
        val_slice_idx=40,
        padding=0,
        lr_flip=0,
        stage2_file=None
    )

    dwi4d_padded = ds.raw_data

    denoised_arr = mppca(dwi4d_padded, patch_radius=args.patch_radius)

    output_path = os.path.join(args.save_dir, "denoised_mppca.npy")
    np.save(output_path, denoised_arr.astype(np.float32))

    print("Saved:", denoised_arr.shape, "to", output_path)

if __name__ == "__main__":
    main()