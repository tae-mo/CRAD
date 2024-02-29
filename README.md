# Grid-Based Continuous Normal Representation for Anomaly Detection

This repository is the official implementation of **Grid-Based Continuous Normal Representation for Anomaly Detection**. 

## Get Started 
### Environment 

**Python3.8**

**Packages**:
- torch==1.12.1
- torchvision==0.13.1

## Requirements
To install requirements:
```setup
pip install -r requirements.txt
```

## Data preparation
1. Download the [**MVTec AD dataset**](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. Construct the data structure as follows:
```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
            |--bottle
            |--cable
            |-- ...
        |-- train.json
        |-- test.json
```

## Training
To train the model(s) in the paper, run this command:
```train
cd experiments/
bash train.sh 4 0,1,2,3 1111
# bash train.sh <num gpus> <gpu ids> <master port>
```

## Evaluation
To evaluate a trained model, run:
```eval
cd experiments/
bash eval.sh 4 0,1,2,3 1111
# bash eval.sh <num gpus> <gpu ids> <master port>
```

## Results
Our model achieves the following performance on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad):

| Platform | GPU | Detection AUROC | Localization AUROC |
| ------ | ------ | ------ | ------ |
| torch.distributed.launch | 4 GPU (NVIDIA RTX A5000 24 GB)|  99.3 | 97.8 |

## BibTeX
```
@article{lee2024grad,
title={Grid-Based Continuous Normal Representation for Anomaly Detection},
author={Lee, Joo Chan and Kim, Taejune and Park, Eunbyung and Woo, Simon S. and Ko, Jong Hwan},
journal={arXiv preprint arXiv:2402.18293},
year={2024}
}
```