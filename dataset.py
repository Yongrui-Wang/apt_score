from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from data import get_kmer_feature


__author__ = "Yongrui Wang"
__license__ = "MIT"

class APTDataset(Dataset):
    def __init__(self):
        file_path = '/home/ibmc-2/my_project/apt/APT220624.R79-AGS.csv'
        f = open(file_path, 'r')
        self.data = []
        for da in f.readlines():
            self.data.append(da.strip('\n').split(','))
        f.close()
        self.one_hot = {'A':torch.tensor([1,0,0,0]), 'T':torch.tensor([0,1,0,0]),
                        'C':torch.tensor([0,0,1,0]), 'G':torch.tensor([0,0,0,1])}


    def __getitem__(self, idx):
        x = {}
        apt_one_hot = torch.tensor([])
        apt = self.data[idx-1][0]
        label = torch.FloatTensor(np.log([float(self.data[idx-1][1])]))
        for nuc in apt:
            apt_one_hot = torch.cat([apt_one_hot, self.one_hot[nuc]], dim=0)
        apt_one_hot = torch.cat([apt_one_hot, apt_one_hot.new_zeros((68-len(apt))*4)])
        apt_one_hot = apt_one_hot.view(68, 4)
        x['one_hot'] = apt_one_hot
        x['kmer'] = get_kmer_feature(self.data[idx-1][0], 4, 68)
        return x, label


    def __len__(self):
        return len(self.data)



