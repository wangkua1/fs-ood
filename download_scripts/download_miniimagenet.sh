#!/usr/bin/env bash
# From https://github.com/renmengye/few-shot-ssl-public
# Download and place "mini-imagenet.tar.gz" in "$DATA_ROOT/mini-imagenet".
DATA_ROOT=data/
mkdir -p $DATA_ROOT/mini-imagenet
cd $DATA_ROOT/mini-imagenet
gdown https://drive.google.com/uc?id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY
tar -xzvf mini-imagenet.tar.gz
rm -f mini-imagenet.tar.gz
