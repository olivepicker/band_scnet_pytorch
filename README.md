## Band-SCNet (wip)

Unofficial implementation proposed [Band-SCNet](https://www.isca-archive.org/interspeech_2025/yang25d_interspeech.pdf) from Yang et al. of Interspeech.

## Citations

```bibtex
@article{yang_band-scnet_nodate,
	title = {Band-{SCNet}: {A} {Causal}, {Lightweight} {Model} for {High}-{Performance} {Real}-{Time} {Music} {Source} {Separation}},
	abstract = {Music source separation (MSS) for real-time applications faces challenges due to significant performance loss. In this paper, we propose Band-SCNet, a lightweight real-time model that bridges the performance gap between real-time and nonreal-time models. Band-SCNet combines Sparse Compression with Cross-band and Narrow-band Blocks to reduce model size and complexity while improving performance. We also introduce the Compressed Self-Attention (CSA) Fusion Module, which enhances efficiency by reducing parameters. Experiments on the MUSDB18-HQ dataset show that Band-SCNet outperforms existing real-time models with an SDR of 7.79 dB, 2.59 million parameters by relaxing to a 92 ms latency, confirming its suitability for real-time MSS. The model achieves a balance between performance, latency, and model size, offering a promising solution for real-time music source separation.},
	language = {en},
	author = {Yang, Junqi and Yang, Yuhong and Tu, Weiping and Zhao, Xin and Lin, Cedar},
}

```