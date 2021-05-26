# This README is heavily outdated and we are planning on releasing a much clearer one, together with pretrained models.

# Contrastive learning methods

Third-party pytorch implementations of contrastive learning methods that supports extra stuff (see "What is available" section).

## Methods available:
* [Barlow Twins](https://arxiv.org/abs/2103.03230)
* [BYOL](https://arxiv.org/abs/2006.07733)
* [MoCo-V2](https://arxiv.org/abs/2003.04297)
* [SimCLR](https://arxiv.org/abs/2002.05709)
* SimCLR + [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
* [SimSiam](https://arxiv.org/abs/2011.10566)
* [Swav](https://arxiv.org/abs/2006.09882)
* [VICReg](https://arxiv.org/abs/2105.04906)
* [NNCLR](https://arxiv.org/abs/2104.14548)

## What is available?
* Online linear evaluation via stop-gradient
* Pytorch-lightning loggining and default benefits (multi-gpu training, mixed precision, etc)
* Gathering negatives across gpu devices to simulate larger batch sizes (gradients don't flow across gpus though)
* Dataloading speed up (at the cost of using more GPU memory) using [Nvidia Dali](https://github.com/NVIDIA/DALI)
* Some methods support multi-resolution crop from [SwAV](https://arxiv.org/abs/2006.09882)
* Post-pretraining linear evaluation (this usually gives 1-1.5% accuracy points)

## TODO
* Update results table
* Run on methods Imagenet
* Implement supervised versions of the algorithms
* Improve Nvidia Dali when using cpu
* Other methods


## Results
| Model    	| Method       	| Dataset      	| Epochs 	| Batch 	| Temperature 	| Projection output 	| Prediction head hidden | Multicrop          	| Dali               	| Supervised         	| Online linear eval 	| Post-pretraining linear eval 	| 
|----------	|--------------	|--------------	|--------	|--------	|-------	|-------------	|-------------------	|--------------------	|--------------------	|--------------------	|--------------------	|------------------------------	|
| Resnet18 	| SimCLR       	| Imagenet-100 	| 100    	| 256   	| 0.2         	| 128               	| :x:   |                      	|                    	|                    	| 70.74              	| 71.02                        	|
| Resnet18 	| SimCLR       	| Imagenet-100 	| 100    	| 256   	| 0.2         	| 128               	| :x: |                    	| :heavy_check_mark: 	|                    	| 70.66              	| 71.64                        	|
| Resnet18 	| SimCLR       	| Imagenet-100 	| 100    	| 256   	| 0.1         	| 128               	| :x: | :heavy_check_mark: 	| :heavy_check_mark: 	|                    	| 73.04              	| 73.72                        	|
| Resnet18 	| SimCLR       	| Imagenet-100 	| 100    	| 256   	| 0.1         	| 128               	| :x: | :heavy_check_mark: 	| :heavy_check_mark: 	| :heavy_check_mark: 	| 85.56              	| 86.16                        	|
| Resnet   	| Barlow Twins 	| Imagenet-100 	| 100    	| 256   	| :x:         	| 2048              	| :x: |                    	| :heavy_check_mark: 	|                    	| 70.72              	| 71.22                        	|
| Resnet   	| SimSiam 	| Imagenet-100 	| 100    	| 256   	| :x:         	| 2048              	| 512 |                    	| :heavy_check_mark: 	|                    	| 66.72              	| 71.16                        	|
| Resnet   	| SimSiam 	| Imagenet-100 	| 100    	| 256   	| :x:         	| 512              	| 128 |                    	| :heavy_check_mark: 	|                    	| 69.28              	| 72.22                        	|
| Resnet   	| SimSiam 	| Imagenet 	| 100    	| 256   	| :x:         	| 512              	| 128 |                    	| :heavy_check_mark: 	|                    	| 55.42              	| ~62.5 atm (running, but strenghting the augmentations seem to have a detrimental effect on imagenet)                         	|

**NOTE:** Heavily outdated. Will be updated soon.

## Notes:
* Barlow Twins and SimCLR work with basically the same hyperparameters
* SimSiam is very hyperparameter-dependent, it can easily collapse with a lr too high and there are a number of tricks that improve performance a lot, e.g. making the lr of the projection head fixed (as described in the paper). Also, the online/offline classifier training doesn't play nice with the same settings as SimCLR.
* SimSiam was tested with 2048 as output for the projection/prediction heads, but maybe it would be better to scale this to 512 (because of the resnet18) and then change the bottleneck on the prediction head from 512 to something like 128. **This is indeed the case.**
* I think that selecting a better lr to the online classifier training for SimSiam can make numbers closer, but I tried with 3.0 (and lower values) and it just converges to a much worse number (this doesn't change the offline linear eval, ofc).

## Requirements
* torch
* tqdm
* wandb
* nvidia-dali
* pytorch-lightning
* lightning-bolts
* einops

**NOTE:** if you are using CUDA 10.X use `nvidia-dali-cuda100` in `requirements.txt`.

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
    --lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_dim 2048 \
    --encoding_dim 2048 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --scale_loss 0.1 \
    --asymmetric_augmentations \
    --name barlow \
    --method barlow_twins \
    --dali \
    --project contrastive_learning \
    --wandb
```
Or follow `bash_files`

**NOTE:** Files try to be up-to-date and follow as closely as possible the recommended parameters of each paper, but check them before running.

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
    --name simclr-linear-eval \
    --pretrained_feature_extractor trained_models/WANDB_RANDOM_ID \
    --project contrastive_learning
```
Or follow `bash_files/run_linear.sh`

**NOTE:** Each method requires their own parameters, so be sure to check their original papers. Although, in our experience this doesn't change so much.
