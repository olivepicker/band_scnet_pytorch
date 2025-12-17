import os
import torch
import librosa as lib
import numpy as np

from einops import rearrange
from torch.utils.data import Dataset

class MUSDBDataset(Dataset):
    def __init__(
        self,
        df,
        is_train=True,
        data_path='data'
    ):
        self.df = df
        self.is_train = is_train
        self.data_path = data_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        path = os.path.join(self.data_path, d.path)
        offset = d.indexs
        
        stem_paths = [path.replace('mixture', n) for n in ['vocals', 'drums', 'bass', 'other']]
        stems = [lib.load(p, sr=44100, mono=False, offset=offset, duration=11)[0] for p in stem_paths]
        stems = torch.tensor(np.array(stems))

        if self.is_train:
            scale = np.random.uniform(0.7, 1.0, (4, 1, 1))
            stems = stems * scale
            mixture = torch.sum(stems, 0)

        else:
            mixture, _ = lib.load(path, sr=44100, mono=False, offset=offset, duration=11)
            mixture = torch.tensor(mixture)

        out = {}
        out['mixture'] = mixture[None,...].float()
        out['stems'] = stems.float()

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