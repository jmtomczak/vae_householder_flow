# Improving Variational Auto-Encoders using Householder Flow
This is a TensorFlow implementation of a new volume-preserving flow using a series of Householder transformations as described in the following paper:
* Jakub M. Tomczak, Max Welling, Improving Variational Auto-Encoders using Householder Flow, NIPS Workshop on Bayesian Deep Learning, [arXiv preprint](https://arxiv.org/abs/1611.09630), 2016

## Data
There are two datasets available:
* MNIST: it will be downloaded automatically;
* Histopathology: before running the experiment this needs to be unpacked.

## Run the experiment
1. Unpack histopathology data.
2. Set-up your experiment in `run_experiment.py` (additional changes could be needed in `VAE_Config.py`).
3. Run experiment:
```bash
python run_experiment.py
```
## Models
You can run a vanilla VAE or a VAE with the Householder Flow (HF) by setting `number_of_Householders` variable to `0` (the vanilla VAE) or `1,2,...` (the VAE+HF).

Additionally, you can choose one of two techniques that highly improve training, namely:
* `warm-up` ([Sønderby, Casper Kaae, et al. "Ladder Variational Autoencoders." NIPS. 2016.](http://papers.nips.cc/paper/6275-ladder-variational-autoencoders.pdf)) ([Bowman, Samuel R., et al. "Generating sentences from a continuous space." arXiv, 2015.](https://arxiv.org/pdf/1511.06349.pdf?TB_iframe=true&width=921.6&height=921.6));
* `free-bits` ([Kingma, Diederik P., et al. "Improving Variational Autoencoders with Inverse Autoregressive Flow", NIPS. 2016](https://papers.nips.cc/paper/6581-improving-variational-autoencoders-with-inverse-autoregressive-flow.pdf)).

## Citation

Please cite our paper if you use this code in your research:

```
@article{TW:2016,
  title={Improving Variational Auto-Encoders using Householder Flow},
  author={Tomczak, Jakub M and Welling, Max},
  journal={arXiv preprint arXiv:1611.09630},
  year={2016}
}
```

## Acknowledgments
The research conducted by Jakub M. Tomczak was funded by the European Commission within the Marie Skłodowska-Curie Individual Fellowship (Grant No. 702666, ”Deep learning and Bayesian inference for medical imaging”).

I am very grateful to [Szymon Zaręba](https://www.ii.pwr.edu.pl/~szymon.zareba/) who prepared the first version of the framework for VAEs using code provided by [Jan Hendrik Metzen](https://jmetzen.github.io/2015-11-27/vae.html). [Szymon](https://www.ii.pwr.edu.pl/~szymon.zareba/) helped me to develop the framework at its early stage.
