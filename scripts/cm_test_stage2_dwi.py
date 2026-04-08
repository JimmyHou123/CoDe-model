#!/usr/bin/env python
import os
import argparse
import torch
import nibabel as nib
import numpy as np
import cv2
import time


from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_noisemodel,
    args_to_dict,
    add_dict_to_argparser,
)
from cm.dwi_datasets import load_data

def _resolve_device(arg_device: str) -> torch.device:
    if arg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg_device)

def _cuda_sync():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()

def _to_u8_minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a float array to uint8 [0,255] for saving PNGs."""
    arr = np.asarray(arr, dtype=np.float32)
    out = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return out

def save_slice_pngs(
    png_root: str,
    slice_idx: int,
    volume_idx: int,
    den_np: np.ndarray,
    noi_np: np.ndarray,
    residual_np: np.ndarray,
    meta: dict,
) -> None:
    """
    Save per-slice outputs in a folder named by slice index.
    Files:
      - volXXX_denoised.png
      - volXXX_noisy.png
      - volXXX_residual.png
      - meta.txt
    """
    slice_dir = os.path.join(png_root, f"slice_{slice_idx:03d}")
    os.makedirs(slice_dir, exist_ok=True)

    base = f"vol{volume_idx:03d}"
    cv2.imwrite(os.path.join(slice_dir, f"{base}_denoised.png"), _to_u8_minmax(den_np))
    cv2.imwrite(os.path.join(slice_dir, f"{base}_noisy.png"),    _to_u8_minmax(noi_np))
    cv2.imwrite(os.path.join(slice_dir, f"{base}_residual.png"), _to_u8_minmax(residual_np))

    meta_path = os.path.join(slice_dir, "meta.txt")
    with open(meta_path, "a") as f:
        f.write(
            "vol={vol} slice={slc} step={step} idx_in_loader={i} sigma_est={sig:.6f} "
            "mask={mask}\n".format(
                vol=volume_idx,
                slc=slice_idx,
                step=meta.get("global_step", 0),
                i=meta.get("i", -1),
                sig=float(meta.get("sigma_est", 0.0)),
                mask=str(meta.get("valid_mask", None)),
            )
        )
        
def save_nifti_volume(denoised_np: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    """Save 4D prediction with dtype, header shape, and qform/sform preserved."""
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_data_shape(denoised_np.shape)

    out_img = nib.Nifti1Image(denoised_np.astype(np.float32, copy=False), ref_img.affine, header=hdr)

    # preserve qform/sform if present
    try:
        qf, qc = ref_img.get_qform(), int(ref_img.header.get("qform_code", 0))
        sf, sc = ref_img.get_sform(), int(ref_img.header.get("sform_code", 0))
        if qf is not None:
            out_img.set_qform(qf, code=qc)
        if sf is not None:
            out_img.set_sform(sf, code=sc)
    except Exception:
        pass

    nib.save(out_img, out_path)
    
def test_dwi(
    model,
    diffusion,
    noise_model,
    loader,
    device,
    ref_img: nib.Nifti1Image,
    valid_mask,                 
    global_step=0,
    save_pngs=False,
    png_root=None,
    save_nifti=True,
    nifti_out_path=None,
):
    H, W, S, V = ref_img.shape
    out_path = None  

    if valid_mask and len(valid_mask) == 2:
        req_start, req_end = int(valid_mask[0]), int(valid_mask[1])
    else:
        req_start, req_end = 0, V

    # Clamp to [0, V]
    mask_start = max(0, min(req_start, V))
    mask_end   = max(0, min(req_end,   V))

    V_mask = max(0, mask_end - mask_start)
    denoised_np = np.zeros((H, W, S, V_mask), dtype=np.float32)

    model.eval()
    noise_model.eval()

    amp_enabled = (device.type == "cuda") and getattr(diffusion, "_amp_enabled", True)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
        for i, batch in enumerate(loader):
            X    = batch["X"].float().to(device)          
            cond = batch["condition"].float().to(device)   
            B = X.shape[0]

            # Stage I
            x0_bar    = noise_model(cond).detach()
            res       = X - x0_bar
            sigma_est = torch.std(res, dim=(1, 2, 3), keepdim=True)
            noise_est = res / (sigma_est + 1e-8)
            noise_mean = torch.mean(noise_est, dim=(1, 2, 3), keepdim=True)

            # Stage II
            sigmas_vec = sigma_est.view(B).to(device)
            denoise_out = diffusion.denoise(model, X, sigmas_vec)[1]
            pred = denoise_out - noise_mean.detach() * sigma_est


            slice_idx = int(batch.get("z_idx", 0))
            n_local   = int(batch.get("n_idx", 0))

            volume_idx = n_local + mask_start

            if not (mask_start <= volume_idx < mask_end):
                if mask_start <= n_local < mask_end:
                    volume_idx = n_local
                else:
                    continue
            
            # write into 4D
            if 0 <= slice_idx < S:
                v_local = volume_idx - mask_start            
                if 0 <= v_local < V_mask:
                    pred_np = pred[0, 0].detach().cpu().numpy().astype(np.float32)
                    denoised_np[..., slice_idx, v_local] = pred_np

                    if save_pngs and png_root:
                        noisy_np    = X[0, 0].detach().cpu().numpy().astype(np.float32)
                        residual_np = (noisy_np - pred_np).astype(np.float32)
                        meta = {
                            "global_step": global_step,
                            "i": i,
                            "sigma_est": float(sigma_est.flatten()[0].detach().cpu().item()),
                            "valid_mask": [mask_start, mask_end],
                        }
                        save_slice_pngs(
                            png_root,
                            slice_idx=slice_idx,
                            volume_idx=volume_idx,   
                            den_np=pred_np,
                            noi_np=noisy_np,
                            residual_np=residual_np,
                            meta=meta,
                        )

    if save_nifti:
        assert nifti_out_path is not None, "nifti_out_path must be set when save_nifti=True"
        os.makedirs(os.path.dirname(nifti_out_path), exist_ok=True)
        save_nifti_volume(denoised_np, ref_img, nifti_out_path)
        out_path = nifti_out_path
        print(f"[INFO] Saved 4D denoised NIfTI: {out_path}")

    if save_pngs and png_root:
        print(f"[INFO] Saved per-slice PNGs + meta under: {png_root}")

    return denoised_np, out_path


def build_argparser():
    """
    CLI for test-time inference:
      - Required: data_root, resume_checkpoint, noisemodel_checkpoint, save_dir
      - Mask:      valid_mask START END  (half-open [START, END))
      - Outputs:   --save_nifti / --save_pngs (proper booleans)
      - Runtime:   batch_size, num_workers, seed, device/AMP
    """
    defaults = model_and_diffusion_defaults()
    parser = argparse.ArgumentParser(
        prog="cm_test_stage2_dwi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_dict_to_argparser(parser, defaults)

    parser.add_argument("--data_root", required=True,
                        help="Path to input 4D DWI NIfTI (.nii/.nii.gz)")
    parser.add_argument("--resume_checkpoint", required=True,
                        help="Stage-2 model checkpoint (.pt/.pth)")
    parser.add_argument("--noisemodel_checkpoint", required=True,
                        help="Noise model checkpoint (.pt/.pth)")
    parser.add_argument("--save_dir", required=True,
                        help="Root directory to write outputs")

    parser.add_argument("--valid_mask", nargs=2, type=int, metavar=("START","END"),
                        default=[10, 160],
                        help="Half-open range [START, END) for global volume indices")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Test-time batch size (script assumes 1 for indexing simplicity)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers")
    parser.add_argument("--rmt_npy_path", type=str, default=None,
                        help="Optional path to RMT numpy file (if your dataset uses it)")

    parser.add_argument("--save_nifti", dest="save_nifti", action="store_true",
                        help="Write the final 4D denoised NIfTI")
    parser.add_argument("--no-save_nifti", dest="save_nifti", action="store_false")
    parser.set_defaults(save_nifti=True)

    parser.add_argument("--save_pngs", dest="save_pngs", action="store_true",
                        help="Write per-slice PNGs (denoised/noisy/residual) + meta")
    parser.add_argument("--no-save_pngs", dest="save_pngs", action="store_false")
    parser.set_defaults(save_pngs=False)

    parser.add_argument("--png_root", type=str, default=None,
                        help="Override PNG output dir; default = <save_dir>/test_imgs")


    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Inference device preference")
    parser.add_argument("--amp", dest="amp", action="store_true",
                        help="Enable autocast mixed precision if supported")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    return parser, defaults


def main():
    parser, defaults = build_argparser()
    args = parser.parse_args()
    ref_img = nib.load(args.data_root)

    device = _resolve_device(args.device)

    md_kwargs = args_to_dict(args, defaults.keys())
    model, diffusion = create_model_and_diffusion(**md_kwargs)
    model.load_state_dict(torch.load(args.resume_checkpoint, map_location="cpu"), strict=True)
    model.to(device).eval()

    unet_args = [
        "image_size", "in_channels", "out_channels", "num_channels",
        "num_res_blocks", "channel_mult", "learn_sigma", "class_cond",
        "use_checkpoint", "attention_resolutions", "num_heads",
        "num_head_channels", "num_heads_upsample",
        "use_scale_shift_norm", "dropout", "resblock_updown",
        "use_fp16", "use_new_attention_order",
    ]
    nm_kwargs = {k: getattr(args, k) for k in unet_args if hasattr(args, k)}
    nm_kwargs["in_channels"] = 2
    noise_model = create_noisemodel(**nm_kwargs)
    noise_model.load_state_dict(torch.load(args.noisemodel_checkpoint, map_location="cpu"), strict=True)
    noise_model.to(device).eval()

    test_loader = load_data(
        dataroot=args.data_root,     
        valid_mask=args.valid_mask,
        phase="test",
        lr_flip=0.0,
        stage2_file=None,
        batch_size=1,
    )

    def _strip_nii_ext(p: str) -> str:
        base = os.path.basename(p)
        if base.endswith(".nii.gz"):
            return base[:-7]
        return os.path.splitext(base)[0]
    
    dataset = _strip_nii_ext(args.data_root)

    method = os.path.basename(os.path.dirname(args.resume_checkpoint))
    if method.startswith("train_hardi_stage2_"):
        method = method[len("train_hardi_stage2_"):]

    ckpt_name = os.path.basename(args.resume_checkpoint)
    step_digits = "".join(ch for ch in ckpt_name if ch.isdigit()) or "0"
    global_step = int(step_digits)

    os.makedirs(args.save_dir, exist_ok=True)
    
    png_root = args.png_root if args.save_pngs else None
    if args.save_pngs and png_root:
        os.makedirs(png_root, exist_ok=True)

    if args.save_nifti:
        nifti_dir = os.path.join(args.save_dir, "nifti")
        os.makedirs(nifti_dir, exist_ok=True)
        nifti_path = os.path.join(
            nifti_dir,
            f"{dataset}_{method}_denoised.nii.gz"
        )
    else:
        nifti_path = None

    H, W, S, V_total = ref_img.shape
    mask_start, mask_end = (args.valid_mask[0], args.valid_mask[1]) if args.valid_mask else (0, V_total)
    V_mask = max(0, min(V_total, mask_end) - max(0, mask_start))
    num_slices = S * V_mask

    _cuda_sync()
    t0 = time.perf_counter()

    setattr(diffusion, "_amp_enabled", bool(args.amp))

    _denoised_np, out_path = test_dwi(
        model=model,
        diffusion=diffusion,
        noise_model=noise_model,
        loader=test_loader,
        device=device,
        ref_img=ref_img,
        valid_mask=args.valid_mask,
        global_step=global_step,
        save_pngs=args.save_pngs,
        png_root=png_root,
        save_nifti=args.save_nifti,
        nifti_out_path=nifti_path,
    )

    _cuda_sync()
    elapsed = time.perf_counter() - t0

    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    time_str = f"{int(hrs):02d}:{int(mins):02d}:{secs:06.3f}"

    throughput_slices = (num_slices / elapsed) if elapsed > 0 else float('nan')
    throughput_vols = (V_mask / elapsed) if elapsed > 0 else float('nan')

    print("[TIMER] Total inference time:", time_str)
    print(f"[TIMER] Processed volumes (mask): {V_mask}  | slices: {num_slices}")
    print(f"[TIMER] Throughput: {throughput_vols:.3f} vols/s, {throughput_slices:.2f} slices/s")

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "timings.txt"), "a") as f:
            f.write(
                f"dataset={dataset} method={method} step={global_step} "
                f"mask=[{mask_start},{mask_end}) "
                f"time_sec={elapsed:.6f} time_hms={time_str} "
                f"throughput_vols_per_s={throughput_vols:.6f} "
                f"throughput_slices_per_s={throughput_slices:.6f}\n"
            )
    except Exception as e:
        print(f"[TIMER] Failed to write timings.txt: {e}")

    if out_path:
        saved = nib.load(out_path)
        print("[DEBUG] Input shape:", ref_img.shape, "Saved shape:", saved.shape)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    main()
    
    
