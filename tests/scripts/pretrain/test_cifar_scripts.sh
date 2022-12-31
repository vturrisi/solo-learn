METHODS=("barlow" "byol" "dino" "mae" "mocov2plus" "mocov3" "nnbyol" "nnclr" "nnsiam" "ressl" "simclr" "simsiam" "supcon" "swav" "vibcreg" "vicreg")
DATASETS=("cifar10")

for dataset in ${DATASETS[@]}; do
    for method in ${METHODS[@]}; do
        echo Running $method
        python3 main_pretrain.py \
            --config-path scripts/pretrain/cifar/ \
            --config-name $method \
            max_epochs=2 \
            devices=[0] \
            optimizer.batch_size=32 \
            ++auto_resume.enabled=False \
            ++wandb.enabled=False \
            ++limit_train_batches=30 \
            ++limit_val_batches=30 \
            ++method_kwargs.warmup_teacher_temperature_epochs=0
        echo ---------------------------------
    done

    # run wmse with larger batch
    method="wmse"
    echo Running $method
    python3 main_pretrain.py \
        --config-path scripts/pretrain/cifar/ \
        --config-name $method \
        max_epochs=2 \
        devices=[0] \
        optimizer.batch_size=128 \
        ++auto_resume.enabled=False \
        ++wandb.enabled=False \
        ++limit_train_batches=30 \
        ++limit_val_batches=30 \
        ++method_kwargs.warmup_teacher_temperature_epochs=0 \
        ++method_kwargs.whitening_size=128
    echo ---------------------------------

    # run deepcluster v2 for a full epoch
    method="deepclusterv2"
    echo Running $method
    python3 main_pretrain.py \
        --config-path scripts/pretrain/cifar/ \
        --config-name $method \
        max_epochs=30 \
        devices=[0] \
        optimizer.batch_size=128 \
        ++auto_resume.enabled=False \
        ++wandb.enabled=False \
        ++limit_val_batches=30 \
        ++method_kwargs.warmup_teacher_temperature_epochs=0 \
        ++dali.encode_indexes_into_labels=True
    echo ---------------------------------

done
