#!/bin/bash

####
#
# Usage: ./prepare_data.sh VAL for using a validation set from the training set. Otherwise use test set.
#
###

MODE=$1

[[ $HOSTNAME == login* ]] || [[ $SLURM_CLUSTER_NAME == mila ]]; mila=$?;

if [ $mila == 0 ];
then
  echo "In Mila - $mila"
  TRAIN=/network/datasets/imagenet/ILSVRC2012_img_train.tar
  VAL=/network/scratch/l/lavoiems/data/imagenet_val.tar
else
  echo "In CC - $mila"
  TRAIN=/network/scratch/lavoiems/data/ILSVRC2012_img_train.tar
  VAL=/network/scratch/lavoiems/data/imagenet_val.tar
fi
TRG=$SLURM_TMPDIR/data
mkdir -p $TRG

tar -xf $TRAIN -C $TRG
if [[ $MODE == "VAL" ]];
then
  mkdir -p $TRG/val
  for D in `ls $TRG/train/`; do
    mkdir -p $TRG/val/$D
    echo ${D}
    ls $TRG/train/$D/ | head -10 | xargs -i mv $TRG/train/$D/{} $TRG/val/$D/
  done
else
  echo "Test"
  tar -xf $VAL -C $TRG
fi
