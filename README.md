# Contrastive learning methods

Third-party pytorch implementations of contrastive learning methods that supports extra stuff (see "What is available" section).

## What is available?
* Contrastive learning pretraining using SimCLR
* Online linear evaluation via stop-gradient
* Pytorch-lightning loggining and default benefits (multi-gpu training, mixed precision, etc)
* Gathering negatives across gpu devices to simulate larger batch sizes (gradients don't flow across gpus though)
* Dataloading speed up (at the cost of using more GPU memory) using [Nvidia Dali](https://github.com/NVIDIA/DALI)
* SimCLR multi-resolution crop from [SwAV](https://arxiv.org/abs/2006.09882)
* SimCLR + [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
* Post-pretraining linear evaluation (this usually gives 1-1.5% accuracy points)

## Working on:
* [Barlow Twins](https://arxiv.org/abs/2103.03230)

## Results
| Model    	| Method 	| Dataset      	| Epochs 	| Batch 	| Temperature 	| Multicrop          	| Dali               	| Supervised         	| Online linear eval 	| Post-pretraining linear eval 	|
|----------	|--------	|--------------	|--------	|-------	|-------------	|--------------------	|--------------------	|--------------------	|--------------------	|------------------------------	|
| Resnet18 	| SimCLR 	| Imagenet-100 	| 100    	| 256   	| 0.2         	|                    	|                    	|                    	| 70.74              	| 71.02                        	|
| Resnet18 	| SimCLR 	| Imagenet-100 	| 100    	| 256   	| 0.2         	|                    	| :white_check_mark: 	|                    	| 70.66              	| 71.64                        	|
| Resnet18 	| SimCLR 	| Imagenet-100 	| 100    	| 256   	| 0.1         	| :white_check_mark: 	| :white_check_mark: 	|                    	| 73.04              	| 73.72                        	|
| Resnet18 	| SimCLR 	| Imagenet-100 	| 100    	| 256   	| 0.1         	| :white_check_mark: 	| :white_check_mark: 	| :white_check_mark: 	| 85.56              	| 86.16                        	|

## Requirements
* torch
* tqdm
* wandb
* nvidia-dali
* pytorch-lightning
* einops

## Installation

```
pip install -r requirements.txt
```

## Pretraining
```
python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --temperature 0.2 \
    --gpus 0 1 \
    --num_workers 8 \
    --hidden_mlp 2048 \
    --no_projection_bn \
    --name simclr \
    --project contrastive_learning
```
Or follow `bash_files/run_contrastive.sh`

## Linear Evaluation
```
python3 main_linear.py \
    imagenet100 \
    resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 10 \
    --no_projection_bn \
    --name simclr-linear-eval \
    --pretrained_feature_extractor trained_models/WANDB_RANDOM_ID \
    --project contrastive_learning
```
Or follow `bash_files/run_linear.sh`