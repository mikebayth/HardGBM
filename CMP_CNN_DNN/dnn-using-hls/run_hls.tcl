open_project rain-proj 
set_top nnet
add_files dnn.cpp
open_solution "solution1"
set_part {xc7vx690tffg1930-1} -tool vivado
create_clock -period 13 -name default
source "directives.tcl"
csynth_design 
exit