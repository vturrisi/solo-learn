python3 ../main_linear.py \
    cifar10 \
    resnet18 \
    --data_folder /data/datasets \
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
    --pretrained_feature_extractor trained_models/PATH \
    --project contrastive_learning
