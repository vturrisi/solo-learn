<!-- Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved -->
<!-- Adapted from https://github.com/facebookresearch/moco/blob/main/detection/README.md -->

## Transferring to Detection

The `train_object_detection.py` script reproduces the object detection experiments on Pascal VOC and COCO.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Convert a pre-trained model to detectron2's format:
   ```
   python3 convert_model_to_detectron2.py --pretrained_feature_extractor PATH_TO_CKPT --output_detectron_model ./detectron_model.pkl
   ```

1. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Run training:
   ```
   python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./detectron_model.pkl
   ```
