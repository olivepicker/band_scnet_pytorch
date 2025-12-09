## Band-SCNet (wip)

Unofficial implementation proposed [Band-SCNet](https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf) from Yang et al. of Interspeech.

## Todo
- [x] Encoder / Decoder baseline
	- [x] Decoder - fix ConvTranspose2d for deconvolution
	- [ ] Decoder - validate crop/pad in SULayer
- [x] Separation Network
- [x] Fusion Network
- [x] Skip Connections
- [ ] Implement Loss
- [ ] Train / Valid pipeline
- [ ] AMC Internal test

## Usage

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