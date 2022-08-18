python3 main_pretrain.py \
    --dataset imagenet \
    --backbone resnet50 \
    --train_data_path /datasets/ILSVRC2012/train \
    --val_data_path /datasets/ILSVRC2012/val \
    --max_epochs 100 \
    --devices 0,1,2,3 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --num_workers 4 \
    --precision 16 \
    --optimizer lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 0.8 \
    --weight_decay 1.5e-6 \
    --batch_size 64 \
    --data_format dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name barlow-resnet50-imagenet-100ep \
    --entity unitn-mhug \
    --project solo-learn \
    --wandb \
    --save_checkpoint \
    --lamb 0.0051 \
    --scale_loss 0.048 \
    --method barlow_twins \
    --proj_hidden_dim 4096 \
    --proj_output_dim 4096 \
    --auto_resume
