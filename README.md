# iwgn_multivariate

This repository runs and produces the results for the panel of experiments in the tabular results of [Importance Weighted Generative Networks](https://arxiv.org/abs/1806.02512).

**Abstract** While deep generative networks can simulate from complex data distributions, their utility can be hindered by limitations on the data available for training. Specifically, the training data distribution may differ from the target sampling distribution due to sample selection bias, or because the training data comes from a different but related distribution. We present methods to accommodate this difference via importance weighting, which allow us to estimate a loss function with respect to a target distribution even if we cannot access that distribution directly. These estimators, which differentially weight the contribution of data to the loss function, offer theoretical guarantees that heuristic approaches lack, while giving impressive empirical performance in a variety of settings.

To run the panel of experiments, use `bash run_panel.sh`. To print results, use `python eval_short_panel.py`.
