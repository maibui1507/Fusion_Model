import os
import argparse
import librosa
import soundfile as sf

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import Dataset,DataLoader, random_split

import numpy as np

___author__ = "Long Nguyen-Vu"
__email__ = "long@ssu.ac.kr"

class ASVDataset(Dataset):
    def __init__(self, protocol_file, dataset_dir, dev = False, eval=False):
        """
        protocol_file: 
            example: `/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/protocol.txt`
            bonafide/LA_T_3424442.wav train - bonafide
            vocoded/hifigan_LA_T_3424442.wav train - spoof
        dataset_dir: directory of the dataset
            example: `/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/`
        """
        self.protocol_file = protocol_file
        self.dataset_dir = dataset_dir
        self.file_list = []
        self.label_list = []
        self.dev = dev  
        self.eval = eval

        with open(self.protocol_file, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                
                # For the case where self.eval is True, ignore the second condition (dev is True)
                if (self.eval and line[1] == "eval") or \
                        (self.dev and line[1] == "dev") or \
                        (line[1] == "train" and line[3] == "bonafide"):
                    self.file_list.append(line[0])
                    self.label_list.append(line[3])

        self._length = len(self.file_list)   
    
    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """return feature and label of each audio file in the protocol file
        """
        audio_file = self.file_list[idx]
        file_path = os.path.join(self.dataset_dir, audio_file)
        
        # Load the audio file with soundfile and convert to tensor
        feature, _ = sf.read(file_path)
        feature_tensors = torch.tensor(feature, dtype=torch.float32)
        
        # Convert label to tensor; 1 if "spoof" else 0
        label = 1 if self.label_list[idx] == "spoof" else 0
        label_tensors = torch.tensor([label], dtype=torch.int64)
        
        return feature_tensors, label_tensors
        
    def collate_fn(self, batch):
        """pad the time series 1D"""
        # pad to max length
        max_width = max(features.shape[0] for features, _ in batch)
        padded_batch_features = []
        for features, _ in batch:
            # Convert to mono if the input is stereo
            if features.ndim > 1 and features.shape[1] == 2:
                features = torch.mean(features, dim=1)  # Average the two channels

            pad_width = max_width - features.shape[0]
            padded_features = F.pad(features, (0, pad_width), mode='constant', value=0)
            padded_batch_features.append(padded_features.unsqueeze(0))
            
        labels = torch.tensor([label for _, label in batch], dtype=torch.int64)
        
        padded_batch_features = torch.cat(padded_batch_features, dim=0)
        return padded_batch_features, labels

    def collate_fn_(self, batch):
        """pad the time series 1D"""
        # pad to max length
        max_width = max(features.shape[0] for features, _ in batch)
        padded_batch_features = []
        for features, _ in batch:
            pad_width = max_width - features.shape[0]
            padded_features = F.pad(features, (0, pad_width), mode='constant', value=0)
            padded_batch_features.append(padded_features)
            
        labels = torch.tensor([label for _, label in batch])
        
        padded_batch_features = torch.stack(padded_batch_features, dim=0)
        return padded_batch_features, labels
    
def get_dataloader(protocol_file, dataset_dir, batch_size, dev=False, eval=False):
    """return dataloader for training and evaluation
    """
    dataset = ASVDataset(protocol_file, dataset_dir, dev=dev, eval=eval)
    if dev or eval:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    return dataloader

def test_dataloader():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_file", type=str, default="/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/protocol.txt")
    ap.add_argument("--dataset_dir", type=str, default="/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    
    print("Test dataloader")
    train_dataloader = get_dataloader(args.protocol_file, args.dataset_dir, args.batch_size, dev=args.dev, eval=args.eval)
    dev_dataloader = get_dataloader(args.protocol_file, args.dataset_dir, args.batch_size, dev=True, eval=args.eval)
    eval_dataloader = get_dataloader(args.protocol_file, args.dataset_dir, args.batch_size, dev=False, eval=True)
    
    # length of train, dev, eval
    print("Length of train, dev, eval")
    print(len(train_dataloader.dataset), len(dev_dataloader.dataset), len(eval_dataloader.dataset))
    