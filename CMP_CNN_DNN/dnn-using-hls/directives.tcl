set_directive_dataflow -disable_start_propagation "nnet"
set_directive_unroll -factor 128 "fc_layer2/fc_layer2_label41"
set_directive_unroll -factor 128 "fc_layer1/fc_layer1_label40"
set_directive_unroll -factor 128 "fc_layer3/fc_layer3_label42"
