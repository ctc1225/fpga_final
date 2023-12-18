#!/bin/bash

MHA_file="MHA_result"
RET_file="RET_result"
n_of_token=("1024" "2048" "3096" "4192" "5120" "6144" "7168")

for n in 0 1 2 3
do 
  echo "run evaluation for $n th time"
  for i in "${n_of_token[@]}"
  do
    echo "python3 mha_compare.py $i, writing to file $MHA_file$n.txt"
    python3 mha_compare.py "$i" >> $MHA_file$n".txt"
    echo "python3 ret_compare.py $i, writing to file $RET_file$n.txt"
    python3 ret_compare.py "$i" >> $RET_file$n".txt"
  done
done