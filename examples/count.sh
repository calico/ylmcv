#!/bin/bash

base_dir=`pwd`
out_dir=`pwd`/examples/output
test_list=$base_dir/examples/test_list.txt
img_dir=$base_dir/examples/data/
num_TTA=3

mkdir -p  $out_dir

python main_predict.py -output_file $out_dir/cnt_output.pkl -num_TTA $num_TTA -model model/model_count.py -checkpoin checkpoint/model_count.pth  -data_dir $img_dir -test_list $test_list -model_out_dim 1 -num_input_channel 1
