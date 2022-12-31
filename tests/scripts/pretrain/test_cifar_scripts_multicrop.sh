METHODS=("swav")
DATASETS=("cifar10")

for dataset in ${DATASETS[@]}; do
    for method in ${METHODS[@]}; do
        echo Running $method
        python3 main_pretrain.py \
            --config-path scripts/pretrain/cifar-multicrop/ \
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
done
