
This github is the implementation of ICLR'22 paper, named as **Coherence-based Label Propagation over Time Series for Accelerated Active Learning**.
Please follow the instructions to reproduce our work.


# Download python packages in requirements.txt
```shell 
pip install -r requirements.txt
```

# Download datasets
Download I3D feature data of 50salads and GTEA at [link](https://zenodo.org/record/3625992#.YVwLbdpBx1N) and locate the contents at the `./datasets/DATASET_NAME`. The link is from [ms-tcn](https://github.com/yabufarha/ms-tcn), the repository for the paper "Y. Abu Farha and J. Gall. MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019." 
For HAPT and mHealth dataset, download [HAPT](http://archive.ics.uci.edu/ml/datasets/smartphone-based+recognition+of+human+activities+and+postural+transitions) and [mHealth](http://archive.ics.uci.edu/ml/datasets/mhealth+dataset) and locate the contents at the `./datasets/DATASET_NAME`

# How to run

At current directory which has all source codes, run main.py with the parameters as follows. 

	- dataset: {50salads, GTEA, mHealth, HAPT} designates which dataset to use.
	- seed: {0, 1, 2, 3, 4} is the seed for 5-fold cross validation.
	- gpu: {0, 1, 2, ...} is an integer for gpu id
	- lp: {platprob, repr, prob, zero} indicates label propagation method to use, representing {TCLP, ESP, PTP, NOP} in the paper, respectively.
	- al: {conf, entropy, margin, core, badge, utility}	shows active learning to use, representing {CONF, ENTROPY, MARG, CS, BADGE, UTILITY} in the paper, respectively.
	- no_plat_reg: {0, 1}	decides whether or not to use width regularization or not. 1 means removing width regularization.
	- temp: [1, infinity] is the parameter T for temperature scaling. T=1 means no temperature scaling.

Here's the example running code.

```shell
python3 main.py --dataset HAPT --gpu 0 --seed 0 --lp platprob --al random --no_plat_reg 1 --temp 2.0
```
Classification accuracy at each active learning round is saved in `metadata` folder as `.npy` format.

# Citation

Please use the following form to cite our paper.

```
@inproceedings{
shin2022coherencebased,
title={Coherence-based Label Propagation over Time Series for Accelerated Active Learning},
author={Yooju Shin and Susik Yoon and Sundong Kim and Hwanjun Song and Jae-Gil Lee and Byung Suk Lee},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=gjNcH0hj0LM}
}
```
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
