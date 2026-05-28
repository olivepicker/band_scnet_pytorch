import os
import torch
import librosa as lib
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset


class MUSDBDataset(Dataset):
    def __init__(
        self,
        df,
        is_train=True,
        data_path="data",
        sr=44100,
        duration=11,
    ):
        self.df = df
        self.is_train = is_train
        self.data_path = data_path
        self.sr = sr
        self.duration = duration
        self.target_length = int(sr * duration)

    def __len__(self):
        return len(self.df)

    def fix_length(self, x):
        length = x.shape[-1]

        if length > self.target_length:
            if self.is_train:
                start = torch.randint(0, length - self.target_length + 1, (1,)).item()
            else:
                start = (length - self.target_length) // 2
            x = x[..., start:start + self.target_length]

        elif length < self.target_length:
            x = F.pad(x, (0, self.target_length - length))

        return x

    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        path = os.path.join(self.data_path, d.path)
        offset = d.indexs

        stem_paths = [
            path.replace("mixture", n)
            for n in ["vocals", "drums", "bass", "other"]
        ]

        stems = [
            lib.load(
                p,
                sr=self.sr,
                mono=False,
                offset=offset,
                duration=self.duration,
            )[0]
            for p in stem_paths
        ]

        stems = torch.tensor(np.array(stems)).float()
        stems = self.fix_length(stems)

        if self.is_train:
            scale = torch.empty(4, 1, 1).uniform_(0.7, 1.0)
            stems = stems * scale
            mixture = torch.sum(stems, dim=0)
        else:
            mixture, _ = lib.load(
                path,
                sr=self.sr,
                mono=False,
                offset=offset,
                duration=self.duration,
            )
            mixture = torch.tensor(mixture).float()
            mixture = self.fix_length(mixture)

        out = {}
        out["mixture"] = mixture[None, ...].float()
        out["stems"] = stems.float()

        return out

'''
# Create index dataframe(overlap 6 seconds)
tmp = pd.DataFrame()
for i in tqdm(range(len(mixtures))):
    wav, sr = lib.load(mixtures[i], mono=False, sr=44100)
    len_sec = wav.shape[1] / 44100
    index_map = np.arange(0, len_sec, 11)

    starts = index_map[:-1].astype(int)
    indexs = np.concatenate([starts, starts + 6])
    
    df = pd.DataFrame(
        {
            'path': mixtures[i],
            'indexs': sorted(indexs)
        }
    )

    tmp = pd.concat([tmp, df])
    ...
'''