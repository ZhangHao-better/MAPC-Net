#!/bin/bash

# 从头训练120k + RTM对比学习
# 配置文件: configs/ascformer/ascformer_rtm_cl_120k.py

export CUDA_VISIBLE_DEVICES=0,1

/home/zhlinux/anaconda3/envs/rtm_linux/bin/python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    tools/train.py \
    configs/ascformer/ascformer_rtm_cl_120k.py \
    --launcher pytorch \
    2>&1 | tee train_cl_120k.log
