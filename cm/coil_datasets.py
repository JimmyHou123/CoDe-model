from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
import pydicom
import glob
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import random
import h5py
import torch.nn.functional as F


def load_data(
    dataroot,
    sequence,
    image_size,
    phase,
    lr_flip,
    first_k,
    stage2_file,
    batch_size
):
    image_dataset = MultiCoilDataset(dataroot = dataroot,
                                     sequence = sequence,
                                     image_size = image_size,
                                     phase = phase,
                                     lr_flip = lr_flip,
                                     first_k = first_k,
                                     stage2_file = stage2_file
                                     )
    loader = DataLoader(dataset = image_dataset,
                        batch_size = batch_size,
                        shuffle = (phase == 'train'),
                        num_workers = 8,
                        drop_last = True
                        )
    return loader

class MultiCoilDataset(Dataset):
    def __init__(self, dataroot, sequence, image_size = (256, 256), phase = 'train', 
                 lr_flip = 0.5, first_k = None, stage2_file = None):
        # sequence: 'T1', 'T2', 'FLAIR'
        self.lr_flip = lr_flip
        self.slice_num = 0
        self.phase = phase
        if sequence:
            files = glob.glob(dataroot + '/*' + sequence + '*')
        else:
            files = glob.glob('*.h5')
        

        if first_k is not None: files = files[: first_k]
        self.X_list = []
        self.condition_list = []
        for file in tqdm(files, desc = 'Loading dataset...'):
            file_temp = h5py.File(file, 'r')
            file_temp.close
            kspace_vol = np.array(file_temp['kspace'])
            images = np.abs(np.fft.ifftshift(np.fft.ifft2(kspace_vol)))
            images = (images - images.min()) / (images.max() - images.min())
            for s in images:
                s = transforms.ToTensor()(s.transpose(1,2,0))
                s = F.interpolate(s.unsqueeze(0), size=image_size, mode='bilinear', align_corners=False).squeeze(0)
                for coil_id, img in enumerate(s):
                    self.X_list.append(2 * img - 1)
                    condition = s.clone()
                    condition[coil_id] = 0
                    self.condition_list.append(2 * condition - 1)

        self.augment = transforms.Compose([
                transforms.RandomVerticalFlip(lr_flip),
                transforms.RandomHorizontalFlip(lr_flip)
            ])   
        
        # running for Stage3?
        if stage2_file is not None:
            print('Parsing Stage2 matched states from the stage2 file...')
            self.matched_state = self.parse_stage2_file(stage2_file)
        else:
            self.matched_state = None
    
    def parse_stage2_file(self, file_path):
        results = dict()
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split('_')
                phase, img_idx, coil_idx, t = info[0], int(info[1]), int(info[2]), int(info[3])
                if phase == self.phase:
                    if img_idx not in results:
                        results[img_idx] = {}
                    results[img_idx][coil_idx] = t
        return results
    
    def __len__(self):
        return len(self.X_list)
    
    def __getitem__(self, idx):
        # raw_input = transforms.ToTensor()(self.img_list[idx].transpose(1,2,0)).unsqueeze(1)
        # raw_input = self.transforms(raw_input).float()
        # X_list, condition_list = [], []
        # for i in range(raw_input.shape[0]):
        #     X = raw_input[i]
        #     condition = torch.cat([raw_input[ : i], raw_input[i + 1 : ]], dim = 0).squeeze(1)
        #     X_list.append(X)
        #     condition_list.append(condition)
        X, condition = self.X_list[idx].unsqueeze(0), self.condition_list[idx]
        if self.phase == 'train':
            combined = torch.cat([X, condition], dim=0) 
            combined = self.augment(combined)
            X, condition = combined[:X.shape[0]], combined[X.shape[0]:]

        ret = dict(X = X, condition = condition)
        # if self.matched_state is not None:
        #     matched_state_list = []
        #     for i in range(raw_input.shape[0]):
        #         state = torch.zeros(1,) + self.matched_state[idx][i]
        #         matched_state_list.append(state)
        #     ret['matched_state'] = torch.stack(matched_state_list, dim = 0)
        return ret #, idx


# dst = MultiCoilDataset(dataroot = '/raid/kaifengpang/M4Raw/multicoil_train', sequence = 'T1', first_k = 5)
# loader = DataLoader(dataset = dst,
#                         batch_size = 2,
#                         shuffle = True,
#                         num_workers = 8,
#                         drop_last = True
#                     )
# for ret, idx in loader:
#     print(ret.shape)
