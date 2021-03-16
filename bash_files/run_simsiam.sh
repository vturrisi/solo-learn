python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.05 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_mlp 2048 \
    --pred_hidden_mlp 512 \
    --encoding_size 2048 \
    --no_projection_bn \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --no_lr_scheduler_for_pred_head \
    --zero_init_residual \
    --name simsiam \
    --method simsiam \
    --dali \
    --project debug
