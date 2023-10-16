#!/bin/bash
image="aisctestacr.azurecr.io/base/job/deepspeed/0.4-pytorch-1.9.0-cuda11.3-a100:20230328T144352732"

docker run \
    --privileged \
    --gpus all \
    -it \
    --ipc=host \
    --network=host \
    --rm \
    -v /drive/ravianupindi/examples:/workspace \
    -v /drive/ravianupindi/imagenet_data:/dataset \
    -w /workspace \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/shadow:/etc/shadow:ro \
    --memory=100g \
    --shm-size=10g \
    $image bash
