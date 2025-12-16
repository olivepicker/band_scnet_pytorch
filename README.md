## Band-SCNet (wip)

Unofficial implementation proposed [Band-SCNet](https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf) from Yang et al. of Interspeech.

## Todo
- [x] Encoder / Decoder baseline
	- [x] Decoder - fix ConvTranspose2d for deconvolution
	- [x] Decoder - validate crop/pad in SULayer
- [x] Separation Network
- [x] Fusion Network
- [x] Skip Connections
- [x] Implement Loss
- [x] Train / Valid pipeline
	- [ ] Metric
- [ ] AMC Internal test

## Usage
```python
from dataset import MUSDBDataset
from band_scnet_pytorch import BandSCNet
from trainer import BandSCNetTrainer

...
train_ds = MUSDBDataset(train_df, is_train=True)
valid_ds = MUSDBDataset(valid_df, is_train=False)

model = BandSCNet(128, enc_in_channels=2, dec_out_channels=8)
trainer = BandSCNetTrainer(
    model=model,
    optimizer=torch.optim.Adam(lr=5e-4, params=m.parameters()),
    batch_size=2,
    train_ds=train_ds,
    valid_ds=valid_ds,
    device='cuda',
    autocast_enabled=False, # autocast not been tested!
)
```

## Citations

```bibtex
@inproceedings{inproceedings,
	author = {Yang, Junqi and Yang, Yuhong and Tu, Weiping and Zhao, Xin and Lin, Cedar},
	year = {2025},
	month = {08},
	pages = {4973-4977},
	title = {Band-SCNet: A Causal, Lightweight Model for High-Performance Real-Time Music Source Separation},
	doi = {10.21437/Interspeech.2025-448}
}

```