python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /data/datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/val \
    --epochs 100 \
    --optimizer sgd \
    --scheduler cosine \
    --lr 0.03 \
    --classifier_lr 0.03 \
    --weight_decay 1e-4 \
    --hidden_dim 512 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 8 \
    --queue_size 65536 \
    --temperature 0.07 \
    --base_tau_momentum 0.999 \
    --final_tau_momentum 0.999 \
    --dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --name mocov2plus \
    --method mocov2plus \
    --project contrastive_learning \
    --wandb
