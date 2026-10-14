`timescale 1ns/1ps
// Independent testbench. The DUT and generated system-IP files are unchanged.
// Only POR, heartbeat and repeat delay COUNTERS are fast-forwarded in the TB;
// core operations, input words, comparisons and handshake cycles are not skipped.
module tb_same_image_system;
  reg sys_clk = 0;
  wire [7:0] led;
  zed_selftest_top dut(.sys_clk(sys_clk), .led(led));
  always #5 sys_clk = ~sys_clk;
  integer cycle = 0;
  integer starts = 0;
  integer completions = 0;
  integer start_cycle = 0;
  integer first_run_start = 0;
  integer last_run_start = 0;
  integer run_count = 0;
  integer comparisons = 0;
  integer input_address_out_of_range_events = 0;
  integer index;
  reg active = 0;
  reg [319:0] written0 = 0;
  reg [319:0] written1 = 0;

  always @(posedge sys_clk) begin
    cycle <= cycle + 1;
    if (dut.ap_start && !active) begin
      starts = starts + 1;
      if ((starts % 2 == 1 && dut.slot30 !== 1'b0) || (starts % 2 == 0 && dut.slot30 !== 1'b1))
        $fatal(1, "Incorrect slot0/slot30 start order");
      start_cycle = cycle;
      active = 1;
      written0 = 0;
      written1 = 0;
      if (!dut.slot30) begin
        last_run_start = cycle;
        if (starts == 1) first_run_start = cycle;
      end
      $display("SYSTEM_START slot=%0d call=%0d cycle=%0d", dut.slot30 ? 30 : 0, starts, cycle);
    end
    if (active && dut.l0_ce && dut.l0_we) begin
      if (dut.l0_addr >= 320) $fatal(1, "Output0 address out of range");
      if (^dut.l0_d === 1'bx) $fatal(1, "Output0 contains X/Z");
      if (dut.l0_d[30:23] == 8'hff) $fatal(1, "Output0 is non-finite");
      written0[dut.l0_addr] = 1;
    end
    if (active && dut.l1_ce && dut.l1_we) begin
      if (dut.l1_addr >= 320) $fatal(1, "Output1 address out of range");
      if (^dut.l1_d === 1'bx) $fatal(1, "Output1 contains X/Z");
      if (dut.l1_d[30:23] == 8'hff) $fatal(1, "Output1 is non-finite");
      written1[dut.l1_addr] = 1;
    end
    if (active && dut.ap_done) begin
      completions = completions + 1;
      active = 0;
      if ((&written0) !== 1'b1 || (&written1) !== 1'b1)
        $fatal(1, "Incomplete 320-word output writes");
      $display("SYSTEM_DONE slot=%0d call=%0d start_to_done_cycles=%0d all_output_words_written=1", dut.slot30 ? 30 : 0, completions, cycle - start_cycle);
    end
    if (!dut.rst && (dut.yre_ce && dut.yre_addr >= 560 || dut.yim_ce && dut.yim_addr >= 560)) begin
      input_address_out_of_range_events = input_address_out_of_range_events + 1;
      if (input_address_out_of_range_events <= 4)
        $display("INPUT_ADDRESS_DIAGNOSTIC cycle=%0d yre_ce=%b yre_addr=%0d yim_ce=%b yim_addr=%0d; unmasked original wrapper read remains unchanged", cycle, dut.yre_ce, dut.yre_addr, dut.yim_ce, dut.yim_addr);
    end
    if ((dut.state == 4'd3 || dut.state == 4'd7) && dut.cmp_ph == 0 && dut.cmp_idx != 0) begin
      comparisons = comparisons + 1;
      if (^dut.l0_q === 1'bx || ^dut.l1_q === 1'bx || dut.last_bit_err === 1'bx)
        $fatal(1, "Unknown comparison output");
    end
    if (cycle > 300000) $fatal(1, "Bounded simulation timeout");
  end

  task inspect_run;
    begin
      wait(dut.state == 4'd8);
      @(negedge sys_clk);
      run_count = run_count + 1;
      if (led[6:0] !== 7'b1100000) $fatal(1, "Unexpected zero-mismatch LED outcome: %b", led);
      if (dut.mm0_r !== 0 || dut.mm_run !== 0) $fatal(1, "Mismatch counter nonzero");
      for (index=0; index<320; index=index+1) begin
        if (^dut.ram_l0[index] === 1'bx || ^dut.ram_l1[index] === 1'bx)
          $fatal(1, "Uninitialized output RAM");
      end
      $display("SYSTEM_RUN_PASS run=%0d slot0_mismatches=0 slot30_mismatches=0 bits=640 start_to_latched_result_cycles=%0d led_low7=%b", run_count, cycle - last_run_start, led[6:0]);
    end
  endtask

  initial begin
    repeat (12) @(negedge sys_clk);
    if (dut.rst !== 1'b1 || dut.ap_start !== 1'b0 || led[6:0] !== 0)
      $fatal(1, "Power-on reset initial behavior failed");
    $display("SYSTEM_RESET_PASS initial_clock_cycles=12 acceleration=counter_terminal_value_only");
    force dut.por_cnt = 22'h3fffff;
    @(negedge sys_clk);
    release dut.por_cnt;
    if (dut.por_done !== 1'b1) $fatal(1, "POR terminal count failed");
    inspect_run;
    force dut.hold_cnt = 28'hfffffff;
    @(negedge sys_clk);
    release dut.hold_cnt;
    inspect_run;
    if (starts != 4 || completions != 4) $fatal(1, "Incorrect start/done count");
    if (comparisons != 1280) $fatal(1, "Incorrect comparison count");
    $display("SYSTEM_REPEAT_PASS normal_runs=2 core_calls=4 comparisons=1280");
    $display("SYSTEM_INTERFACE_DIAGNOSTIC input_address_out_of_range_events=%0d output_unknown_events=0 comparator_unknown_events=0 no_unknowns_masked=1", input_address_out_of_range_events);

    $display("SYSTEM_INTEGRATED_PASS image_source=local_a12_fabric normal_comparisons=1280 physical_board_test=0 counter_delays_accelerated=1");
    $finish;
  end
endmodule
