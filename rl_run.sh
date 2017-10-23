#!/bin/bash

PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH

python3.6 oscar/RlDqnAgent.py --mode train --memory-size 40000 --model Neuralnetwork/DenseMineralShard.knn --outmodel Neuralnetwork/DenseMineralShard.knn --trainstep 300000
