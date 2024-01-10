import os
import glob
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ActionData(Dataset):
    def __init__(self, root, data_type, neighbors, augmentation=None,
                 num_frames=50, stride=50, neighbor_pattern='nearest'):

        self.root = root
        self.split_data = pd.read_csv(os.path.join(root, data_type + '.csv'), header=None, delimiter=',')
        self.split_data = self.split_data.drop(0)

        speaker_path = [path for path in list(self.split_data.values[:, 1])]
        listener_path = [path for path in list(self.split_data.values[:, 2])]

        self.neighbors = neighbors                  # None
        self.aug = augmentation                     # None
        self.num_frames = num_frames                # 50
        self.stride = stride                        # 25
        self.neighbor_pattern = neighbor_pattern    # 'all'
        self.all_data = self.get_site_id_clip(speaker_path, listener_path, data_type)

    def get_site_id_clip(self, speaker_path, listener_path, data_type):
        all_data = []

        for index in range(len(speaker_path)):
            site, group, pid, clip = speaker_path[index].split('/')
            all_data.extend([data_type, site, group, pid, clip, str(i)]
                            for i in range(0, 750 - self.num_frames + 1, self.stride))
            
            site, group, pid, clip = listener_path[index].split('/')
            all_data.extend([data_type, site, group, pid, clip, str(i)]
                            for i in range(0, 750 - self.num_frames + 1, self.stride))

        return all_data

    def __getitem__(self, index):
        dtype_site_group_pid_clip_idx = self.all_data[index]

        v_inputs = self.load_video_pth(dtype_site_group_pid_clip_idx)
        a_inputs = self.load_audio_pth(dtype_site_group_pid_clip_idx)

        if self.neighbor_pattern in {'nearest', 'all'}:
            # target:   'train+NoXI+065_2016-04-14_Nottingham+speaker+1+0'
            #           'dtype+site+group+pid+clip+idx'
            targets = '+'.join(dtype_site_group_pid_clip_idx)
        
        return v_inputs, a_inputs, targets

    def __len__(self):
        return len(self.all_data)

    def load_video_pth(self, dtype_site_group_pid_clip_idx):
        dtype, site, group, pid, clip, idx = dtype_site_group_pid_clip_idx
        idx = int(idx)
        video_pth = os.path.join(self.root, dtype, 'Video_features', site, group, pid, clip + '.pth')
        video_inputs = torch.load(video_pth, map_location='cpu')[idx:idx + self.num_frames]
        return video_inputs

    def load_audio_pth(self, dtype_site_group_pid_clip_idx):
        dtype, site, group, pid, clip, idx = dtype_site_group_pid_clip_idx
        idx = int(idx)
        audio_pth = os.path.join(self.root, dtype, 'Audio_features', site, group, pid, clip + '.pth')
        audio_inputs = torch.load(audio_pth, map_location='cpu')[idx:idx + self.num_frames]
        if audio_inputs.shape[0] != self.num_frames:
            audio_inputs = torch.cat([audio_inputs, audio_inputs[-1].unsqueeze(dim=0)])
        return audio_inputs