# python3 ../main_contrastive.py \
#     imagenet100 \
#     resnet18 \
#     --data_folder /data/datasets \
#     --train_dir imagenet-100/train \
#     --val_dir imagenet-100/test \
#     --epochs 100 \
#     --optimizer sgd \
#     --scheduler warmup_cosine \
#     --lr 0.3 \
#     --classifier_lr 30.0 \
#     --weight_decay 1e-4 \
#     --batch_size 128 \
#     --gpus 0 1 \
#     --num_workers 4 \
#     --hidden_mlp 2048 \
#     --pred_hidden_mlp 512 \
#     --encoding_size 2048 \
#     --no_projection_bn \
#     --zero_init_residual \
#     --name simsiam \
#     --method simsiam \
#     --dali \
#     --project debug

# --no_lr_scheduler_for_pred_head \

python3 ../main_linear.py \
    imagenet100 \
    resnet18 \
    --data_folder /data/datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --scheduler step \
    --lr 30.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 10 \
    --no_projection_bn \
    --name simsiam-linear-eval \
    --pretrained_feature_extractor trained_models/188qi01q \
    --dali \
    --project debug

# python3 ../main_contrastive.py \
#     imagenet100 \
#     resnet18 \
#     --data_folder /data/datasets \
#     --train_dir imagenet-100/train \
#     --val_dir imagenet-100/test \
#     --epochs 200 \
#     --optimizer sgd \
#     --scheduler warmup_cosine \
#     --lr 0.3 \
#     --classifier_lr 3.0 \
#     --weight_decay 1e-4 \
#     --batch_size 128 \
#     --gpus 0 1 \
#     --num_workers 4 \
#     --hidden_mlp 2048 \
#     --pred_hidden_mlp 512 \
#     --encoding_size 2048 \
#     --no_projection_bn \
#     --zero_init_residual \
#     --name simsiam \
#     --method simsiam \
#     --dali \
#     --project debug

python3 ../main_linear.py \
    imagenet100 \
    resnet18 \
    --data_folder /data/datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --scheduler step \
    --lr 30.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 10 \
    --no_projection_bn \
    --name simsiam-linear-eval \
    --pretrained_feature_extractor trained_models/b65x5xu5 \
    --dali \
    --project debug
