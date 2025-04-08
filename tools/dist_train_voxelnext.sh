#!/bin/bash

CONFIG="configs/mvxnet/mvxnet_voxelnext_train.py"
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --launcher pytorch \
    --cfg-options env_cfg.dist_cfg.port=$PORT \
    ${@:3}

# Optional flags:
# --resume-from <checkpoint>
# --auto-scale-lr
# --no-validate
