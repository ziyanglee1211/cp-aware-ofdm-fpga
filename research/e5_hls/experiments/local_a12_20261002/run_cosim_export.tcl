open_project proj_a12
set_top receiver_slot_opt
add_files -tb tb_a12_cosim.cpp
open_solution sol_opt
cosim_design -rtl verilog -trace_level none
export_design -format ip_catalog -rtl verilog
exit
