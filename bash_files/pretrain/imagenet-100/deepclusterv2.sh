python3 ../../../main_pretrain.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_dir /data/datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/val \
    --max_epochs 400 \
    --gpus 0,1 \
    --distributed_backend ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.6 \
    --min_lr 0.0006 \
    --warmup_start_lr 0.0 \
    --warmup_epochs 11 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 4 \
    --dali \
    --encode_indexes_into_labels \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --name deepclusterv2-400ep-imagenet100 \
    --entity unitn-mhug \
    --project solo-learn \
    --wandb \
    --method deepclusterv2 \
    --wandb \
    --proj_hidden_dim 2048 \
    --proj_output_dim 128 \
    --num_prototypes 3000 3000 3000
