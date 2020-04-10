#!/usr/bin/env bash

echo "generating dataset..."
dataset=$1

cd $dataset
rm -rf valid/ train/ train.txt valid.txt
cd -
python part.py $2 $dataset
