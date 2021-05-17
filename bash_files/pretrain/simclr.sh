python3 ../main_contrastive.py \
    --dataset imagenet100 \
    --encoder resnet18 \
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
    --num_workers 8 \
    --name simclr \
    --project contrastive_learning \
    --wandb \
    simclr \
    --temperature 0.2 \
    --hidden_dim 2048

python3 ../main_contrastive.py \
    --dataset imagenet100 \
    --encoder resnet18 \
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
    --name simclr-dali \
    --dali \
    --project contrastive_learning \
    --wandb \
    simclr \
    --temperature 0.2 \
    --hidden_dim 2048

python3 ../main_contrastive.py \
    --dataset imagenet100 \
    --encoder resnet18 \
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
    --multicrop \
    --n_crops 2 \
    --n_small_crops 6 \
    --name multi-crop-simclr-dali \
    --dali \
    --project contrastive_learning \
    --wandb \
    simclr \
    --hidden_dim 2048 \
    --temperature 0.1

python3 ../main_contrastive.py \
    --dataset imagenet100 \
    --encoder resnet18 \
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
    --multicrop \
    --supervised \
    --n_crops 2 \
    --n_small_crops 6 \
    --name multi-crop-supervised-simclr-dali \
    --dali \
    --project contrastive_learning \
    --wandb \
    simclr \
    --temperature 0.1 \
    --hidden_dim 2048
