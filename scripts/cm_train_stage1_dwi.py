"""
Train a diffusion model on images.
"""

import argparse

from cm import dist_util, logger
from cm.dwi_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_noisemodel,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import NoiseModelTrainLoop
import torch.distributed as dist
import copy
import torch, random
import numpy as np
import warnings
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
torch.manual_seed(42)
random.seed(0)
np.random.seed(0)
torch.cuda.set_device(0)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir = args.save_dir)

    logger.log("creating model...")

    model = create_noisemodel(    
                        image_size = args.image_size,
                        in_channels = args.in_channels,
                        out_channels = args.out_channels,
                        num_channels = args.num_channels,
                        num_res_blocks = args.num_res_blocks,
                        channel_mult = args.channel_mult,
                        use_checkpoint=False,
                        attention_resolutions="16",
                        num_head_channels=args.num_head_channels,
                        use_scale_shift_norm=args.use_scale_shift_norm,
                        dropout=args.dropout,
                        resblock_updown=args.resblock_updown,
                        use_fp16=args.use_fp16
                        )
    model.to(dist_util.dev())
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    train_data = load_data(
        dataroot = args.data_root,
        valid_mask = [129, 193],
        phase = 'train',
        lr_flip = 0.5,
        stage2_file = None,
        batch_size = batch_size
    ) # 11400 samples

    val_data = load_data(
        dataroot = args.data_root,
        valid_mask = [129, 193],
        phase = 'val',
        lr_flip = 0,
        stage2_file = None,
        batch_size = 1
    )


    logger.log("training...")
    NoiseModelTrainLoop(
        model=model,
        total_training_steps=args.total_training_steps,
        data=train_data,
        val_data = val_data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        epoch = args.epoch
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_root = "",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        log_interval=10,
        save_interval=1,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        save_dir="",
        epoch = 20,
        channel_mult = (1, 2, 4, 8, 16),
        in_channels = 4,
        out_channels = 1,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
