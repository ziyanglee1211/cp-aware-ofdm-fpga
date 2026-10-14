# Same wrapper/device/clock/implementation flow as rev-T; new A12 IP and golden.
set root "E:/biyelunwen/e5_hls/experiments/local_a12_20261002/board"
set iprepo "E:/biyelunwen/e5_hls/experiments/local_a12_20261002/ip_a12_fabric"
if {[file exists "$root/proj/zed_selftest.xpr"]} {error "Preserve existing project: do not overwrite"}
if {![file exists "$iprepo/component.xml"]} {error "A12 IP missing"}
create_project zed_selftest "$root/proj" -part xc7z020clg484-1
set_property ip_repo_paths $iprepo [current_project]
update_ip_catalog
create_ip -vlnv xilinx.com:hls:receiver_slot_opt:1.0 -module_name dccn_hls_ip
generate_target all [get_ips dccn_hls_ip]
create_ip_run [get_ips dccn_hls_ip]
add_files -fileset sources_1 "$root/rtl/zed_selftest_top.v"
add_files -fileset sources_1 "$root/rtl/pred_roms.vh"
set_property file_type {Verilog Header} [get_files pred_roms.vh]
add_files -fileset constrs_1 "$root/constrs/zed_selftest.xdc"
set_property top zed_selftest_top [current_fileset]
update_compile_order -fileset sources_1
set iprun [get_runs dccn_hls_ip_synth_1]
if {[llength $iprun] != 1} {error "IP OOC synthesis run missing"}
set_property STEPS.SYNTH_DESIGN.ARGS.MAX_DSP 200 $iprun
set_property STEPS.SYNTH_DESIGN.ARGS.MAX_DSP 220 [get_runs synth_1]
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {error "Synthesis failed"}
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {error "Implementation failed"}
open_run impl_1
report_timing_summary -file "$root/reports/system_timing_summary.rpt"
report_utilization -file "$root/reports/system_utilization.rpt"
report_power -file "$root/reports/system_power.rpt"
report_drc -file "$root/reports/system_drc.rpt"
report_route_status -file "$root/reports/system_route_status.rpt"
puts "LOCAL_A12_BOARD_BUILD_COMPLETE"
exit
