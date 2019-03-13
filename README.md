# Multi-Inference-GAN

## Zhanfu Yang, Purdue University
This is a similation codes I submmited in the paper to the ICML 2019.

## 3 Domain

x,y,z Transfermation

To Run:

`cd our3domain`

`bash train.sh`

## 2 Domain Step 1
To Run:

`cd 2domain_1`

`bash train.sh`
## 2 Doamin Step 2
To Run:

`cd 2domain_2`

`bash train.sh`
## Parameters

`--dataset_size_x` Datasets size of the distribution x.

`--dataset_size_x_size` Test Datasets size of the distribution x.

`----load_model` To load the number of model

`--gpu_id` The execution of the GPU

`--model_path` The path of the Model

`--lr_xz` The initial of XZ Learning rate

