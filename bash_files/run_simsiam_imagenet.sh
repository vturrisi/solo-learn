python3 ../main_contrastive.py \
    imagenet100 \
    resnet50 \
    --data_folder /datasets \
    --train_dir ILSVRC2012/train \
    --val_dir ILSVRC2012/val \
    --epochs 100 \
    --optimizer sgd \
    --scheduler cosine \
    --lr 0.05 \
    --classifier_lr 30.0 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_dim 2048 \
    --pred_hidden_dim 512 \
    --encoding_dim 2048 \
    --zero_init_residual \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --name simsiam-resnet50-imagenet-same-parameters \
    --method simsiam \
    --no_lr_scheduler_for_pred_head \
    --dali \
    --project contrastive_learning
