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
    --classifier_lr 30.0 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_dim 2048 \
    --pred_hidden_dim 128 \
    --encoding_dim 512 \
    --zero_init_residual \
    --name simsiam-smaller-heads \
    --method simsiam \
    --no_lr_scheduler_for_pred_head \
    --dali \
    --project contrastive_learning

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
    --classifier_lr 30.0 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 4 \
    --hidden_dim 2048 \
    --pred_hidden_dim 512 \
    --encoding_dim 2048 \
    --zero_init_residual \
    --name simsiam \
    --method simsiam \
    --no_lr_scheduler_for_pred_head \
    --dali \
    --project contrastive_learning
