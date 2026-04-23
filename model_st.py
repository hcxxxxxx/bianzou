import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn.functional as F
import os
# from dinat import DinatLayer1d, DinatLayer2d
from tqdm import tqdm
sr = 44100
hop_length = 512
frame_duration = hop_length / sr  # ≈ 0.0116秒



#==========
import math
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable
import torch.nn as nn
from natten.functional import natten1dav, natten1dqkrpb, natten2dav, natten2dqkrpb


# ==== 数据准备 ====

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os
import random



sr = 44100
hop_length = 512
frame_duration = hop_length / sr  # ≈ 0.0232秒

# ==== 数据准备 ====

class ChromaDataset(torch.utils.data.Dataset):
    def __init__(self, label_dict, chroma_dir, sr=44100, hop_length=512, tolerance=3,fole_time = 1):
        self.file_list = list(label_dict.keys())
        self.label_dict = label_dict
        self.chroma_dir = chroma_dir
        self.sr = sr
        self.hop_length = hop_length
        self.tolerance = tolerance
        self.fole_time = fole_time
    def __len__(self):
        return len(self.file_list)

    def generate_frame_labels(self, time_points, n_frames):
        frame_labels = np.zeros(n_frames, dtype=np.float32)
        frame_duration = self.hop_length / self.sr
        for t in time_points:
            center = int(t / frame_duration)
            delta = int(self.tolerance / frame_duration)
            start = max(center - delta, 0)
            end = min(center + delta + 1, n_frames)
            frame_labels[start:end] = 1.0
        return frame_labels

    
    def fold_sequence(self, seq, fold_size):
        n = seq.shape[0]
        n_folds = n // fold_size
        seq = seq[:n_folds * fold_size]  # truncate to multiple
        seq = seq.reshape(n_folds, fold_size, -1).mean(axis=1)
        return seq

    def __getitem__(self, idx):
        file_id = self.file_list[idx].replace('.wav','')
        chroma_path = os.path.join(self.chroma_dir, file_id + ".npy")
        chroma = np.load(chroma_path)  # (12, T)
        time_points = self.label_dict[self.file_list[idx]]

        n_frames = chroma.shape[1]
        labels = self.generate_frame_labels(time_points, n_frames)  # (T,)

        # 转置 chroma: (T, 12)
        chroma = chroma.T
        frame_duration = self.hop_length / self.sr
        fold_frames = int(self.fole_time / frame_duration)

        # === 折叠 chroma 和 labels ===
        # folded_chroma = self.fold_sequence(chroma, fold_frames)     # shape: (T', 12)
        folded_labels = self.fold_sequence(labels[:, None], fold_frames).squeeze()  # shape: (T',)

        chroma = torch.tensor(chroma, dtype=torch.float32)
        folded_labels = torch.tensor(folded_labels, dtype=torch.float32)

        return chroma, folded_labels, time_points


class AllInOneSectionOnly(nn.Module):
    def __init__(self,fole_time = 1,dim_embed = 24,lstm_hidden_size = 128, lstm_num_layers = 2):
        super().__init__()
        self.num_levels = 6
        self.num_features = int(dim_embed * 2 ** (self.num_levels - 1))
        self.embeddings = AllInOneEmbeddings(dim_embed)
       
        self.norm = nn.LayerNorm(dim_embed, eps=1e-5)
        self.dropout = nn.Dropout(0.1)
        self.fold_size =  int(fole_time / frame_duration)
        self.section_classifier = Head(hs = 256, num_classes=1, init_confidence=0.001)
        self.lstm = nn.LSTM(
            input_size=dim_embed*self.fold_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
    def fold_tensor_expand_dim(self, x, fold_size):
        batch, T, C = x.shape
        n_fold = T // fold_size
        x = x[:, :n_fold * fold_size, :]  # 截断多余帧
        x = x.view(batch, n_fold, fold_size, C)  # (batch, n_fold, fold_size, C)
        return x
    def forward(self, inputs: torch.FloatTensor, output_attentions: Optional[bool] = None):
        T,  F = inputs.shape
        N = 1
        K =1
        inputs = inputs.reshape(-1, 1, T, F)  # N*K, 1, T, F
      
        fold_size = self.fold_size
        frame_embed = inputs.view(1,T, F)
        folded = self.fold_tensor_expand_dim(frame_embed, fold_size)  # (N*K, n_fold, fold_size, C)
        batch, n_fold, fold_size, C = folded.shape
        folded_reshaped = folded.view(batch * n_fold, fold_size* C)  # (batch*n_fold, fold_size, C)
        lstm_out, _ = self.lstm(folded_reshaped)  # shape: N*K, T, 2*dim_embed
        hidden_states = lstm_out.reshape(N, K, n_fold, -1)  # N, K, T, C
        logits_section = self.section_classifier(hidden_states).squeeze()  # N, T
        
        return logits_section


class AllInOneEmbeddings(nn.Module):
    def __init__(self,dim_embed=24):
        super().__init__()
        dim_input, hidden_size = 81, dim_embed

        self.act_fn = nn.ELU()
        first_conv_filters = hidden_size // 2  # 硬编码
        drop_conv = 0.2

        self.conv0 = nn.Conv2d(1, first_conv_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.pool0 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.drop0 = nn.Dropout(drop_conv)

        self.conv1 = nn.Conv2d(first_conv_filters, hidden_size, kernel_size=(1, 12), stride=(1, 1), padding=(0, 0))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=(0, 0))
        self.drop1 = nn.Dropout(drop_conv)

        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # 保证输出 F=1

        self.norm = nn.LayerNorm(dim_embed)
        self.dropout = nn.Dropout(drop_conv)

    def forward(self, x: torch.FloatTensor):

        x = self.conv0(x)
        x = self.pool0(x)
        x = self.act_fn(x)
        x = self.drop0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.act_fn(x)
        x = self.drop1(x)

        x = self.conv2(x)        

        x = self.pool2(x)
        x = self.act_fn(x)
        embeddings = x.squeeze(-1)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

