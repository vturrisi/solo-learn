python3 main_pretrain.py \
    --dataset imagenet100 \
    --backbone vit_small \
    --data_dir $1 \
    --train_h5_path train.h5 \
    --val_h5_path val.h5 \
    --max_epochs 400 \
    --warmup_epochs 40 \
    --devices 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer adamw \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 3.0e-4  \
    --classifier_lr 3.0e-4 \
    --weight_decay 0.1 \
    --batch_size 64 \
    --num_workers 8 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --min_scale 0.08 \
    --num_crops_per_aug 1 1 \
    --name mocov3-vit-400ep-imagenet100 \
    --project solo-learn \
    --entity unitn-mhug \
    --save_checkpoint \
    --wandb \
    --method mocov3 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0
