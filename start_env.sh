#!/bin/bash
image="aisctestacr.azurecr.io/base/job/deepspeed/0.4-pytorch-1.9.0-cuda11.3-cudnn8-devel:20230328T144353812"

[[ -z $1 ]] && exit

docker run \
    --privileged \
    --gpus all \
    --name $1 \
    -it \
    --ipc=host \
    --rm \
    -v /home/ravianupindi/examples:/workspace \
    -v /home/ravianupindi/imagenet_data:/dataset \
    -v /home/ravianupindi/checkpoints/:/checkpoints \
    -w /workspace \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/shadow:/etc/shadow:ro \
    --memory=100g \
    --shm-size=10g \
    $image bash
