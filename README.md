# Few-shot Out-of-Distribution Detection
This repository contains code used for 
```
Kuan-Chieh Wang, Paul Vicol, Eleni Triantafillou, Richard Zemel. 
(2020). “Few-shot Out-of-Distribution Detection.” 
International Conference on Machine Learning (ICML) 
Workshop on Uncertainty and Robustness in Deep Learning
```


## Python Environment

Python==3.7.3
For packages, see `requirements.txt`.

Example cmd for creating a Conda Env
```
conda create -n pnl python=3.7 pip
source activate pnl
pip install -r requirements.txt
```


## Download Datasets

**Download miniImageNet**
```bash
./download_scripts/download_miniimagenet.sh
```


To download the out-of-dataset (OOS) datasets, run:
```bash
./download_scripts/download_anomaly.sh  # Downloads the Texture, Places, and notMNIST datasets
./download_scripts/download_isun.sh
./download_scripts/download_lsun.sh
./download_scripts/download_tinyimagenet.sh
```


## Experiments

### (Pre-)Training backbone/encoder

**To train standard (all-way) classifiers**  
for CIFAR-FS
```bash
./run_scripts/classify_classic/cifar-submitted.sh
```
for miniImageNet
```bash
./run_scripts/classify_classic/miniimagenet-submitted.sh
```

### Training OEC for different ways/shots  
```bash
./run_scripts/train_confidence/ooe-submitted.sh
```


### Evaluate OOD   
for CIFAR-FS
```bash
./run_scripts/eval_ood/submitted.sh
```


### Train ProtoNet and MAML 
```bash
./run_scripts/train/submitted.sh
```


## Acknowledgement
This code is built upon many other repos.   
An incomplete list includes:
* https://github.com/jakesnell/prototypical-networks
* https://github.com/wyharveychen/CloserLookFewShot
* https://github.com/hendrycks/outlier-exposure

Also, during development of the private (and ugly repo), my colleagues [Paul](https://github.com/asteroidhouse) and [Eleni](https://github.com/eleniTriantafillou) contributed in very significant ways.

## Citation

If you use this code, consider citing:

```
@article{wang2020fsood,
  title={Few-shot Out-of-Distribution Detection},
  author={Kuan-Chieh Wang and Paul Vicol and Eleni Triantafillou and Richard Zemel},
  booktitle={{International Conference on Machine Learning (ICML) Workshop on Uncertainty and Robustness in Deep Learning}},
  year={2020}
}
```
