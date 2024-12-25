# OnlineTSF
![Stars](https://img.shields.io/github/stars/SJTU-DMTai/OnlineTSF)
[![Visits Badge](https://badges.pufler.dev/visits/SJTU-DMTai/OnlineTSF)](https://badges.pufler.dev/visits/SJTU-Quant/OnlineTSF)

This is an online time series forecasting framework empowered by online model adaptation (i.e., continual model updates).

This repo also contains the official code of PROCEED, a novel online method proposed in [Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting](https://arxiv.org/pdf/2412.08435) (Research Track in KDD 2025).

## 🚀Outstanding Features
### No Information Leakage
Our key observation is that,
the ground truth of each time series sample is not available until after the forecast horizon.
However, most existing online learning implementations involve information leakage as 
they update models with the latest forecasted sample. 

Thus, we timely provide this framework that always update the model with the observed ground truth.

### Supported Methods
- **Proceed**: Proactive Model Adaptation. [[KDD 2025]](https://arxiv.org/pdf/2412.08435) [[Code]](https://github.com/SJTU-DMTai/OnlineTSF)
- **SOLID**: Sample-level Contextualized Adapter. [[KDD 2024]](https://arxiv.org/abs/2310.14838) [[Code]](https://github.com/HALF111/calibration_CDS)
- **OneNet**: Online Ensembling Network. [[NeurIPS 2024]](https://arxiv.org/abs/2309.12659) [[Code]](https://github.com/yfzhang114/OneNet)
- **FSNet**: Learning Fast and Slow. [[ICLR 2023]](https://openreview.net/forum?id=RNMIIZwzCg) [[Code]](https://github.com/salesforce/fsnet)
- **DER++**: Dark Experience Replay. [[NeurIPS 2020]](https://arxiv.org/abs/2004.07211) [[Code]](https://github.com/aimagelab/mammoth)
- **ER**: Experience Replay.
- **Naive**: Online Gradient Descent.

### Fair Experiments and Easier Development
Another observation is that most methods conduct experiments in isolated settings,
where the optimizer and other training details may differ from each other. 
Also, their codes are of low reusability as all operations mixed together.
It would be hard for the followers to keep fair comparison by carefully checking their coding details.

In this framework, we make an endeavor to improve the code structure. 
We develop some basic functions and override them with existing methods,
making the method differences more clear for the readers now.

It would also be easier for the followers to explore different experiment settings by modifying the common functions shared by all methods.

### Full Data Utilization
To mitigate data hungry and concept drift, 
we involve the validation data in online learning.
It is also noteworthy that the common practice leaves out several time series samples to
keep no overlap between the training & validation & test data. 
Assuming that online learning is a continual process, we also take these samples for updates.


## Usage
### Scripts
Please refer to the directory `scripts`. 

First run scripts for pretraining; then run scripts for online learning.

### Arguments
Basic arguments:
- `--model` decides the forecast backbone.
- `--seq_len` decides the lookback length.
- `--pred_len` decides the forecast horizon.
- `--dataset` decides the dataset of which the file path is configured in `settings.py`.
- `--learning_rate` controls the learning rate when training on historical data.
- `--online_learning_rate`: controls the learning rate when training on online data.
> By default, the argument `--border_type` is set to `online` for a data split of 20:5:75 for training/validation/test.
> Empirically, we can set a higher value of `online_learning_rate` than the value tuned on the validation data, 
> since the online phase is much longer. 
> In practice, we can periodically tune the online learning rate on new recent data, instead of keeping a fixed one.

Hyperparameters of Proceed:
- `--concept_dim` corresponds to $d_c$ in our paper.
- `--bottleneck_dim` corresponds to $r$ in our paper.


## Citation
If you find this repo useful, please consider citing:
```
@InProceedings{Proceed,
  author       = {Lifan Zhao and Yanyan Shen},
  booktitle    = {Proceedings of the 31st {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  title        = {Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting},
  year         = {2025},
  month        = {feb},
  publisher    = {{ACM}},
  doi          = {10.1145/3690624.3709210},
}
```