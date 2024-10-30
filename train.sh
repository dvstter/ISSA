#!/bin/zsh

for n in {1..3}
do
  python imitationtrain.py seed=$n && ./clear_all.sh
done