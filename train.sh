#!/bin/zsh

for n in {1..5}
do
  python piegtrain.py seed=$n && ./clear_all.sh
done