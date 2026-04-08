from curses import raw
from torch.utils.data import Dataset, DataLoader
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import random
import os
import numpy as np
import torch
from dipy.io.image import save_nifti, load_nifti
from matplotlib import pyplot as plt
from torchvision import transforms, utils

def load_data(
    dataroot,
    valid_mask,
    phase,
    image_size = 128,
    in_channel = 1,
    val_volume_idx = 40,
    val_slice_idx = 40,
    padding = 3,
    lr_flip = 0.5,
    stage2_file = None,
    batch_size = 1,
    rmt_npy_path=None
):
    image_dataset = MRIDataset(dataroot = dataroot,
                               valid_mask = valid_mask,
                               phase = phase,
                               image_size = image_size,
                               in_channel=in_channel, 
                               val_volume_idx=val_volume_idx, 
                               val_slice_idx=val_slice_idx,
                               padding=padding, 
                               lr_flip=lr_flip, 
                               stage2_file = stage2_file,
                               rmt_npy_path = rmt_npy_path
                               )
    loader = DataLoader(dataset = image_dataset,
                        batch_size = batch_size,
                        shuffle = (phase == 'train'),
                        num_workers = 8
                        )
    return loader



class MRIDataset(Dataset):
    def __init__(self, 
                 dataroot, 
                 valid_mask, 
                 phase='train', 
                 image_size=128, 
                 in_channel=1, 
                 val_volume_idx=50, 
                 val_slice_idx=40,
                 padding=1, 
                 lr_flip=0.5, 
                 stage2_file=None,
                 rmt_npy_path=None):
        self.padding = padding // 2
        self.lr_flip = lr_flip
        self.phase = phase
        self.in_channel = in_channel
        
        # for rmt denoise
        self.rmt_npy_path = rmt_npy_path
        self._rmt_mem = None  # lazy memmap

        # read data
        raw_data, _ = load_nifti(dataroot) # width, height, slices, gradients
        print('Loaded data of size:', raw_data.shape)
        # normalize data
        raw_data = raw_data.astype(np.float32) / np.max(raw_data, axis=(0,1,2), keepdims=True)

        # parse mask
        assert type(valid_mask) is (list or tuple) and len(valid_mask) == 2
 
        # mask data
        raw_data = raw_data[:,:,:,valid_mask[0]:valid_mask[1]] 
        self.data_size_before_padding = raw_data.shape
        print("masked:", raw_data.shape)

        self.raw_data = np.pad(raw_data.astype(np.float32), ((0,0), (0,0), (in_channel//2, in_channel//2), (self.padding, self.padding)), mode='wrap').astype(np.float32)
        print("padded:", self.raw_data.shape)
        
        # running for Stage3?
        if stage2_file is not None:
            print('Parsing Stage2 matched states from the stage2 file...')
            self.matched_state = self.parse_stage2_file(stage2_file)
        else:
            self.matched_state = None

        # transform
        if phase == 'train':
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Resize(image_size),
                transforms.RandomVerticalFlip(lr_flip),
                transforms.RandomHorizontalFlip(lr_flip),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Resize(image_size),
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])

        # prepare validation data
        if val_volume_idx == 'all':
            self.val_volume_idx = range(raw_data.shape[-1])
        elif type(val_volume_idx) is int:
            self.val_volume_idx = [val_volume_idx]
        elif type(val_volume_idx) is list:
            self.val_volume_idx = val_volume_idx
        else:
            self.val_volume_idx = [int(val_volume_idx)]

        if val_slice_idx == 'all':
            self.val_slice_idx = range(0, raw_data.shape[-2])
        elif type(val_slice_idx) is int:
            self.val_slice_idx = [val_slice_idx]
        elif type(val_slice_idx) is list:
            self.val_slice_idx = val_slice_idx
        else:
            self.val_slice_idx = [int(val_slice_idx)]
        
        # if self.phase == 'test':
            # all_volume_idx = list(range(self.data_size_before_padding[-1]))  # volume indices
            # all_slice_idx = list(range(self.data_size_before_padding[-2]))   # slice indices
            # all_indices = [(v, s) for v in all_volume_idx for s in all_slice_idx]
            # random.seed(42)  # for reproducibility
            # sampled = random.sample(all_indices, min(100, len(all_indices)))  # max 100 slices

            # override __len__ and __getitem__ logic with sampled list
            # self.test_index_list = sampled  # store tuples of (volume_idx, slice_idx)

    # for rmt denoise
    def _lazy_load_rmt(self):
        if self.rmt_npy_path is not None and self._rmt_mem is None:
            # Shape (H, W, D, N); must match ds.raw_data shape
            self._rmt_mem = np.load(self.rmt_npy_path, mmap_mode='r')


    def parse_stage2_file(self, file_path):
        results = dict()
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                info = line.strip().split('_')
                volume_idx, slice_idx, t = int(info[0]), int(info[1]), int(info[2])
                if volume_idx not in results:
                    results[volume_idx] = {}
                results[volume_idx][slice_idx] = t
        return results


    def __len__(self):
        if self.phase == 'train' or self.phase == 'test':
            return self.data_size_before_padding[-2] * self.data_size_before_padding[-1] # num of volumes
        elif self.phase == 'val':
            return len(self.val_volume_idx) * len(self.val_slice_idx)
    # def __len__(self):
        # if self.phase == 'test' and hasattr(self, 'test_index_list'):
            # return len(self.test_index_list)
        # elif self.phase == 'train' or self.phase == 'test':
            # return self.data_size_before_padding[-2] * self.data_size_before_padding[-1]
        # elif self.phase == 'val':
            # return len(self.val_volume_idx) * len(self.val_slice_idx)

    # def __getitem__(self, index):
    #     if self.phase == 'train' or self.phase == 'test':
    #         # decode index to get slice idx and volume idx
    #         volume_idx = index // self.data_size_before_padding[-2]
    #         slice_idx = index % self.data_size_before_padding[-2]
    #     elif self.phase == 'val':
    #         s_index = index % len(self.val_slice_idx)
    #         index = index // len(self.val_slice_idx)
    #         slice_idx = self.val_slice_idx[s_index]
    #         volume_idx = self.val_volume_idx[index]

    #     raw_input = self.raw_data
       
    #     if self.padding > 0:
    #         raw_input = np.concatenate((
    #                                 raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,volume_idx:volume_idx+self.padding],
    #                                 raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,volume_idx+self.padding+1:volume_idx+2*self.padding+1],
    #                                 raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding]]), axis=-1)

    #     elif self.padding == 0:
    #         raw_input = np.concatenate((
    #                                 raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding-1]],
    #                                 raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding]]), axis=-1)

    #     # w, h, c, d = raw_input.shape
    #     # raw_input = np.reshape(raw_input, (w, h, -1))
    #     if len(raw_input.shape) == 4:
    #         raw_input = raw_input[:,:,0]
    #     raw_input = self.transforms(raw_input) # only support the first channel for now
    #     # raw_input = raw_input.view(c, d, w, h)

    #     ret = dict(X=raw_input[[-1], :, :], condition=raw_input[:-1, :, :])

    #     if self.matched_state is not None:
    #         ret['matched_state'] = torch.zeros(1,) + self.matched_state[volume_idx][slice_idx]

    #     return ret
    
    # For rmt denoise
    def __getitem__(self, index):
        if self.phase == 'train' or self.phase == 'test':
            # decode index to get slice idx and volume idx
            volume_idx = index // self.data_size_before_padding[-2]
            slice_idx  = index %  self.data_size_before_padding[-2]
        elif self.phase == 'val':
            s_index    = index % len(self.val_slice_idx)
            index      = index // len(self.val_slice_idx)
            slice_idx  = self.val_slice_idx[s_index]
            volume_idx = self.val_volume_idx[index]
    
        raw_input = self.raw_data
    
        if self.padding > 0:
            raw_input = np.concatenate((
                raw_input[:, :, slice_idx:slice_idx + 2 * (self.in_channel // 2) + 1, volume_idx:volume_idx + self.padding],
                raw_input[:, :, slice_idx:slice_idx + 2 * (self.in_channel // 2) + 1, volume_idx + self.padding + 1:volume_idx + 2 * self.padding + 1],
                raw_input[:, :, slice_idx:slice_idx + 2 * (self.in_channel // 2) + 1, [volume_idx + self.padding]]
            ), axis=-1)
    
        elif self.padding == 0:
            raw_input = np.concatenate((
                raw_input[:, :, slice_idx:slice_idx + 2 * (self.in_channel // 2) + 1, [volume_idx + self.padding - 1]],
                raw_input[:, :, slice_idx:slice_idx + 2 * (self.in_channel // 2) + 1, [volume_idx + self.padding]]
            ), axis=-1)
    
        # Keep your original 4D -> 3D reduction
        if len(raw_input.shape) == 4:
            raw_input = raw_input[:, :, 0]
    
        # Append the precomputed RMT slice as an extra channel BEFORE transforms
        if self.rmt_npy_path is not None:
            if self._rmt_mem is None:
                self._rmt_mem = np.load(self.rmt_npy_path, mmap_mode="r")  # shape (H,W,D,N_padded)
            # Match the center direction in the padded axis
            x_rmt_hw = self._rmt_mem[:, :, slice_idx, volume_idx]  # (H,W)
            raw_input = np.concatenate([raw_input, x_rmt_hw[..., None]], axis=-1)  # last channel now X_rmt
    
        # Apply your transforms to ALL channels together
        chw = self.transforms(raw_input)  # -> [C_total(+1 if RMT), H, W]
    
        # Split outputs
        if hasattr(self, "rmt_npy_path") and (self.rmt_npy_path is not None):
            # Channels order entering transforms: [..., center, X_rmt]
            X_rmt = chw[[-1], :, :]       
            X     = chw[[-2], :, :]        
            condition = chw[:-2, :, :]     
            ret = dict(X=X, X_rmt=X_rmt, condition=condition)
        else:
            # Original behavior (no RMT)
            ret = dict(X=chw[[-1], :, :], condition=chw[:-1, :, :])
    
        if self.matched_state is not None:
            ret['matched_state'] = torch.zeros(1,) + self.matched_state[volume_idx][slice_idx]
    
        # Expose indices for debugging/alignment
        ret['z_idx'] = slice_idx
        ret['n_idx'] = volume_idx
    
        return ret