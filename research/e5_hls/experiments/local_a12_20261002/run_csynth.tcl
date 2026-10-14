# Same tool, device, clock and interfaces as rev-T; independent project only.
open_project -reset proj_a12
set_top receiver_slot_opt
add_files receiver_local_a12.cpp
open_solution -reset sol_opt -flow_target vivado
set_part {xc7z020clg484-1}
create_clock -period 10 -name default
set_directive_interface -mode ap_memory -depth 560 receiver_slot_opt y_re
set_directive_interface -mode ap_memory -depth 560 receiver_slot_opt y_im
set_directive_interface -mode ap_memory -depth 320 receiver_slot_opt logit0
set_directive_interface -mode ap_memory -depth 320 receiver_slot_opt logit1
csynth_design
exit
