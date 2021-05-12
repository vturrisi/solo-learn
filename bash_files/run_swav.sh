python3 ../main_contrastive.py \
    cifar10 \
    resnet18 \
    --data_folder ../datasets \
    --epochs 1000 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.6 \
    --classifier_lr 0.03 \
    --weight_decay 1e-6 \
    --batch_size 256 \
    --gpus 0 \
    --num_workers 8 \
    --hidden_dim 2048 \
    --queue_size 3840 \
    --epoch_queue_starts 100 \
    --name swav \
    --method swav \
    --project contrastive_learning \
    --wandb
