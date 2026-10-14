# ZedBoard (xc7z020clg484-1) constraints - PL-only self-test
# Pins per Avnet ZedBoard master XDC Rev C/D v2
# (https://gist.github.com/tsotnep/35c4266f36abecfc0c5b).

# 100 MHz oscillator (GCLK)
set_property -dict { PACKAGE_PIN Y9  IOSTANDARD LVCMOS33 } [get_ports sys_clk]
create_clock -period 10.000 -name sys_clk [get_ports sys_clk]

# LEDs LD7..LD0
set_property -dict { PACKAGE_PIN T22 IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
set_property -dict { PACKAGE_PIN T21 IOSTANDARD LVCMOS33 } [get_ports {led[1]}]
set_property -dict { PACKAGE_PIN U22 IOSTANDARD LVCMOS33 } [get_ports {led[2]}]
set_property -dict { PACKAGE_PIN U21 IOSTANDARD LVCMOS33 } [get_ports {led[3]}]
set_property -dict { PACKAGE_PIN V22 IOSTANDARD LVCMOS33 } [get_ports {led[4]}]
set_property -dict { PACKAGE_PIN W22 IOSTANDARD LVCMOS33 } [get_ports {led[5]}]
set_property -dict { PACKAGE_PIN U19 IOSTANDARD LVCMOS33 } [get_ports {led[6]}]
set_property -dict { PACKAGE_PIN U14 IOSTANDARD LVCMOS33 } [get_ports {led[7]}]

# config DRC
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

# timing exceptions: none (single clock domain, all paths timed at 100 MHz)
