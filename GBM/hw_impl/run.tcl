create_project mypro -part xc7vx690tffg1930-1       

set outputDir ./vivado_out
set treeDir ./trees
file mkdir $outputDir

read_verilog [glob ./*.v]
read_verilog [glob $treeDir/*.v]
read_xdc top.xdc

create_run synth -flow {Vivado Synthesis 2018}
create_run impl -flow {Vivado Implementation 2018} -parent_run synth
launch_run synth impl -jobs 32

wait_on_run impl
open_run impl

report_utilization -file $outputDir/place_utilization.rpt
report_power -file $outputDir/place_power.rpt