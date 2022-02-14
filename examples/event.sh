#!/bin/bash

base_dir=`pwd`
out_dir=`pwd`/examples/output
test_list=$base_dir/examples/test_list.txt
img_dir=$base_dir/examples/data/
num_TTA=3

mkdir -p  $out_dir


# predicting using the first model
python main_predict.py -output_file $out_dir/event_output_01.pkl -num_TTA $num_TTA -model model/model_event01.py -checkpoin checkpoint/model_event01.pth  -data_dir $img_dir -test_list $test_list -model_out_dim 3 -num_input_channel 1

# predicting using the second model
python main_predict.py -output_file $out_dir/event_output_02.pkl -num_TTA $num_TTA -model model/model_event02.py -checkpoin checkpoint/model_event02.pth  -data_dir $img_dir -test_list $test_list -model_out_dim 3 -num_input_channel 1
