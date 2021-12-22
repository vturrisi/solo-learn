#!/bin/bash
DATA_PATH=$1
ROOT_PATH=$2
DATASET=$3
EXPNAME=byol-resnet50-$DATASET

orion -vv hunt -n $EXPNAME --config=../../../orion_config.yaml \
    python3 ../../../main_pretrain.py \
    --wandb \
    --name="${EXPNAME}-{trial.hash_params}" \
    --group=${EXPNAME} \
    --entity il_group  \
    --project VIL \
    --save_checkpoint \
    --checkpoint_dir="${ROOT_PATH}/{trial.hash_params}" \
    --auto_resume \
    --data_dir=${DATA_PATH} \
    --train_dir ILSVRC2012/train \
    --val_dir ILSVRC2012/val \
    --max_epochs~'fidelity(low=100,high=1000,base=4)' \
    --dataset imagenet \
    --backbone resnet50 \
    --num_workers=${SLURM_CPUS_PER_TASK} \
    --gpus 0,1 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.45 \
    --accumulate_grad_batches 16 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 4 \
    --dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --method byol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier
