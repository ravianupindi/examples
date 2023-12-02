#!/bin/bash

[[ -z $1 ]] && exit

filename="$1"
batch_times=`grep -i "Batch:" $filename | grep "Rank: 0" | cut -d" " -f15`
avg_batch_time=$(echo $batch_times | tr ' ' \\n | awk '{ if ($0 != 0) { s += $0; ++total_batches; } } END { print s/total_batches  }')
echo $avg_batch_time
