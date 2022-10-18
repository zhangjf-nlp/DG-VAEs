# DG-VAE

This is the pytorch implementation of methods and experiments introduced in our paper *Improving Variational Autoencoders with Density Gap-based Regularization*.

## Acknowledgements

Thanks for sharing the code to public! A large portion of this repo is borrowed from https://github.com/valdersoul/bn-vae.

## Requirements

* Jupyter Notebook >= 6.1.4
* Python >= 3.6
* PyTorch >= 1.6.0
* GPUtil
* rouge

## Quick Start

1. Download data through **prepare_data_yelp_yahoo.py** and **prepare_data.py**
2. (optional) Modify **args_settings.py** to define which datasets and which models you want to take experiments on
3. Train models through running **one_key_experiment.ipynb** in Jupyter Notebook
4. (optional) Duplicate **one_key_experiment.ipynb** and run the copies in parallel (synchronized by file system, not completely safe)
5. Monitor the training procedure and evaluate models through running **watch_experiments.ipynb** in Jupyter Notebook
6. Visualize the latent space for models through running **latent_space_visualization.ipynb** in Jupyter Notebook

## Project Outline

```python
.
""" code for data downloading and encapsulation (borrowed from previous work) """
+--data
| +--...
+--config
| +--...
+--prepare_data.py
+--prepare_data_yelp_yahoo.py

""" code for experiments """
+--args_settings.py             # defines which datasets and which models to take experiments on
+--experiment.py                # defines mainly the training process (partly borrowed from previous work)
+--load_and_test_models.py      # defines mainly the evaluation process
+--utils.py                     # for functions used in evaluation
+--utils_old.py                 # for functions used in evaluation (borrowed from previous work)
+--cuda_utils.py                # for dynamic GPU allocation
+--local_info.py                # for environment information recording

""" notebooks for efficiently taking experiments """
+--one_key_experiment.ipynb     # running this notebook, you can take experiments on all settings defined
                                # in args_settings.py by iteration (each experiment takes one GPU);
                                # running copies of this notebook, you can take those experiments in parallel
                                # (each copy of this notebook takes one GPU).
+--watch_experiments.ipynb      # running this notebook, you can watch/monitor the procedure of those experiments;
                                # when all those experiments are complete, evaluation for them will start, the
                                # results of which will be written into excel files.
+--latent_space_visualization.ipynb
                                # for latent space visualization
+--toy_experiment.ipynb         # for toy experiments on a synthetic dataset

""" code for methods definition """
+--modules
| +--decoders
| | +--decoder.py               # skip-VAE, bow-VAE (partly borrowed from previous work)
| | +--decoder_helper.py
| | +--__init__.py
| +--encoders
| | +--encoder.py               # VAE, BN-VAE, Delta-VAE, FB-VAE, vMF-VAE, AAE, WAE, DG-VAE and DG-vMF-VAE
| | +--DGBasedKLD.py            # implementation of Eq. 9 for DG-vMF-VAE, and Eq. 11 for DG-VAE
| | +--utils.py                 # implementation of KLD, PDF and sampling for Gaussian distribution and vMF distribution
| | +--__init__.py
| +--utils.py
| +--vae.py                     # defines the framework of VAE for text generation (partly borrowed from previous work)
| +--__init__.py

""" BLEU, Self-BLEU, Distinct and Rouge """
+--nlg_metrics
| +--...

""" this file """
+--README.md
```