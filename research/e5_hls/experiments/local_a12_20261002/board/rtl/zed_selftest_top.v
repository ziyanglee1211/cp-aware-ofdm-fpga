`timescale 1ns / 1ps
//
// zed_selftest_top - PL-only bring-up self-test for the local A12 receiver
// on Avnet ZedBoard (xc7z020clg484-1), Vivado 2022.2.
//
// After configuration the design waits ~42 ms (POR), then runs a two-slot
// regression: slot 0 (ETU 12 dB) and slot 30 (ETU 24 dB) of stim.h, the same
// vectors tb_fft_cosim.cpp uses. Inputs come from corrected-PDP distributed-ROM
// copies of the stimulus (little BRAM18 is free, so nothing can go to
// block RAM), the classifier hard decision (l1 > l0, exact IEEE-754 compare
// via lexicographic key transform) is compared against PyTorch FFT golden
// vector, and the outcome shows on LD7..LD0.  The regression repeats every
// ~2.7 s; any failure latches sticky on LD4.
//
// LED map (ZedBoard LD7..LD0):
//   LD7       ~1 Hz heartbeat (clock alive)
//   LD6       at least one full two-slot run completed (no timeout)
//   LD5       LAST run passed (zero mismatches on BOTH slots)
//   LD4       sticky FAIL (any run failed or timed out since POR)
//   LD3..LD0  last-run total mismatch (slot0 + slot30), saturates at 15
//
// Expected steady state (RTL co-sim: mismatch 0/320 on both slots):
//   LD7 blinking, LD6 on, LD5 on, LD4 off, LD3..LD0 = 0000.
//
module zed_selftest_top (
    input  wire       sys_clk,   // 100 MHz oscillator (pin Y9)
    output wire [7:0] led        // LD7..LD0
);

    // ------------------------------------------------------------------
    // clock / power-on reset (no external reset button needed)
    // ------------------------------------------------------------------
    wire clk = sys_clk;

    reg [21:0] por_cnt  = 22'd0;
    reg        por_done = 1'b0;
    always @(posedge clk) begin
        if (!por_done) begin
            por_cnt <= por_cnt + 22'd1;
            if (&por_cnt)                 // 2^22 cycles = 41.9 ms
                por_done <= 1'b1;
        end
    end
    wire rst = ~por_done;                 // active-high, feeds ap_rst

    // ------------------------------------------------------------------
    // stimulus ROMs - slot 0 (12 dB) / slot 30 (24 dB), 560 x float32
    // ------------------------------------------------------------------
    localparam integer SAMPLES = 560;     // FF*SS
    localparam integer OUTN    = 320;     // DD

    (* rom_style = "distributed" *) reg [31:0] rom_yre_s0  [0:SAMPLES-1];
    (* rom_style = "distributed" *) reg [31:0] rom_yim_s0  [0:SAMPLES-1];
    (* rom_style = "distributed" *) reg [31:0] rom_yre_s30 [0:SAMPLES-1];
    (* rom_style = "distributed" *) reg [31:0] rom_yim_s30 [0:SAMPLES-1];

    initial begin
        $readmemh("E:/biyelunwen/e5_hls/experiments/local_a12_20261002/board/data/yre_s0.mem",  rom_yre_s0);
        $readmemh("E:/biyelunwen/e5_hls/experiments/local_a12_20261002/board/data/yim_s0.mem",  rom_yim_s0);
        $readmemh("E:/biyelunwen/e5_hls/experiments/local_a12_20261002/board/data/yre_s30.mem", rom_yre_s30);
        $readmemh("E:/biyelunwen/e5_hls/experiments/local_a12_20261002/board/data/yim_s30.mem", rom_yim_s30);
    end

    // golden hard decisions (independent PyTorch FFT), 320 bits per slot
    `include "pred_roms.vh"

    // ------------------------------------------------------------------
    // HLS IP instance (receiver_slot_opt local A12, 2022.2 export)
    // ------------------------------------------------------------------
    wire [9:0]  yre_addr, yim_addr;
    wire        yre_ce,   yim_ce;
    reg  [31:0] yre_q,    yim_q;
    wire [8:0]  l0_addr,  l1_addr;
    wire        l0_ce, l0_we, l1_ce, l1_we;
    wire [31:0] l0_d, l1_d;
    wire        ap_start, ap_done, ap_idle, ap_ready;

    dccn_hls_ip u_ip (
        .ap_clk          (clk),
        .ap_rst          (rst),
        .ap_start        (ap_start),
        .ap_done         (ap_done),
        .ap_idle         (ap_idle),
        .ap_ready        (ap_ready),
        .y_re_address0   (yre_addr),
        .y_re_ce0        (yre_ce),
        .y_re_q0         (yre_q),
        .y_im_address0   (yim_addr),
        .y_im_ce0        (yim_ce),
        .y_im_q0         (yim_q),
        .logit0_address0 (l0_addr),
        .logit0_ce0      (l0_ce),
        .logit0_we0      (l0_we),
        .logit0_d0       (l0_d),
        .logit1_address0 (l1_addr),
        .logit1_ce0      (l1_ce),
        .logit1_we0      (l1_we),
        .logit1_d0       (l1_d)
    );

    // slot select + stimulus read (sync read, as HLS ap_memory expects)
    reg slot30 = 1'b0;
    always @(posedge clk) begin
        if (yre_ce) yre_q <= slot30 ? rom_yre_s30[yre_addr] : rom_yre_s0 [yre_addr];
        if (yim_ce) yim_q <= slot30 ? rom_yim_s30[yim_addr] : rom_yim_s0 [yim_addr];
    end

    // ------------------------------------------------------------------
    // output capture RAMs, 320 x float32 (distributed: no BRAM left)
    // ------------------------------------------------------------------
    (* ram_style = "distributed" *) reg [31:0] ram_l0 [0:OUTN-1];
    (* ram_style = "distributed" *) reg [31:0] ram_l1 [0:OUTN-1];

    always @(posedge clk) begin
        if (l0_ce & l0_we) ram_l0[l0_addr] <= l0_d;
        if (l1_ce & l1_we) ram_l1[l1_addr] <= l1_d;
    end

    // ------------------------------------------------------------------
    // hard-decision compare datapath (all inputs registered: short paths)
    //   bit_q == (float(l1_q) > float(l0_q)) exactly, via lexicographic key
    // ------------------------------------------------------------------
    reg  [31:0] l0_q, l1_q;
    reg         pred_q;
    wire [31:0] k0 = l0_q[31] ? ~l0_q : (l0_q ^ 32'h8000_0000);
    wire [31:0] k1 = l1_q[31] ? ~l1_q : (l1_q ^ 32'h8000_0000);
    wire        bit_q = (k1 > k0);
    wire        last_bit_err = (bit_q != pred_q);

    // ------------------------------------------------------------------
    // self-test FSM
    // ------------------------------------------------------------------
    localparam [3:0] S_START0  = 4'd0,
                     S_RUN0    = 4'd1,
                     S_DRAIN0  = 4'd2,
                     S_CMP0    = 4'd3,
                     S_START30 = 4'd4,
                     S_RUN30   = 4'd5,
                     S_DRAIN30 = 4'd6,
                     S_CMP30   = 4'd7,
                     S_HOLD    = 4'd8;

    // Board-level equivalence is stricter than a BER acceptance test: the
    // exported RTL must reproduce the two co-simulation vectors bit-for-bit.
    localparam [8:0] PASS_TH = 9'd0;
    localparam [8:0] OUTN_9  = 9'd320;    // OUTN as a 9-bit compare value
    localparam [19:0] TIMEOUT  = 20'hF_FFFF;  // 1.05M cycles = 10.5 ms >> 58k

    reg [3:0]  state   = S_START0;
    reg        ap_start_r = 1'b0;
    reg [19:0] tout_cnt   = 20'd0;
    reg [8:0]  cmp_idx    = 9'd0;
    reg        cmp_ph     = 1'b0;
    reg [8:0]  mm_run     = 9'd0;         // mismatches, current slot
    reg [8:0]  mm0_r      = 9'd0;         // mismatches, slot 0 (latched)
    reg [27:0] hold_cnt   = 28'd0;

    reg        done_ever   = 1'b0;
    reg        pass_last   = 1'b0;
    reg        fail_sticky = 1'b0;
    reg [3:0]  mm_last     = 4'd0;

    wire [9:0] mm_total = mm0_r + mm_run + last_bit_err;  // at CMP30 end

    always @(posedge clk) begin
        if (!por_done) begin
            state      <= S_START0;
            ap_start_r <= 1'b0;
        end else begin
            case (state)
            // ---------------- slot 0 (ETU 12 dB) ----------------
            S_START0: begin
                slot30 <= 1'b0;
                if (ap_idle) begin
                    ap_start_r <= 1'b1;
                    tout_cnt   <= 20'd0;
                    state      <= S_RUN0;
                end
            end
            S_RUN0: begin
                ap_start_r <= 1'b0;
                if (ap_done) begin
                    state <= S_DRAIN0;
                end else begin
                    tout_cnt <= tout_cnt + 20'd1;
                    if (tout_cnt == TIMEOUT) begin
                        fail_sticky <= 1'b1;
                        state       <= S_HOLD;
                        hold_cnt    <= 28'd0;
                    end
                end
            end
            S_DRAIN0: begin
                if (ap_idle) begin
                    cmp_idx <= 9'd0;
                    cmp_ph  <= 1'b0;
                    mm_run  <= 9'd0;
                    state   <= S_CMP0;
                end
            end
            S_CMP0: begin
                if (cmp_ph == 1'b0) begin
                    if (cmp_idx != 9'd0)
                        mm_run <= mm_run + (last_bit_err ? 9'd1 : 9'd0);
                    if (cmp_idx == OUTN_9) begin
                        mm0_r <= mm_run + (last_bit_err ? 9'd1 : 9'd0);
                        state <= S_START30;
                    end else begin
                        cmp_ph <= 1'b1;
                    end
                end else begin
                    l0_q   <= ram_l0[cmp_idx];
                    l1_q   <= ram_l1[cmp_idx];
                    pred_q <= slot30 ? PRED_S30[cmp_idx] : PRED_S0[cmp_idx];
                    cmp_idx <= cmp_idx + 9'd1;
                    cmp_ph  <= 1'b0;
                end
            end
            // ---------------- slot 30 (ETU 24 dB) ----------------
            S_START30: begin
                slot30 <= 1'b1;
                if (ap_idle) begin
                    ap_start_r <= 1'b1;
                    tout_cnt   <= 20'd0;
                    state      <= S_RUN30;
                end
            end
            S_RUN30: begin
                ap_start_r <= 1'b0;
                if (ap_done) begin
                    state <= S_DRAIN30;
                end else begin
                    tout_cnt <= tout_cnt + 20'd1;
                    if (tout_cnt == TIMEOUT) begin
                        fail_sticky <= 1'b1;
                        state       <= S_HOLD;
                        hold_cnt    <= 28'd0;
                    end
                end
            end
            S_DRAIN30: begin
                if (ap_idle) begin
                    cmp_idx <= 9'd0;
                    cmp_ph  <= 1'b0;
                    mm_run  <= 9'd0;
                    state   <= S_CMP30;
                end
            end
            S_CMP30: begin
                if (cmp_ph == 1'b0) begin
                    if (cmp_idx != 9'd0)
                        mm_run <= mm_run + (last_bit_err ? 9'd1 : 9'd0);
                    if (cmp_idx == OUTN_9) begin
                        // both slots done: latch run result
                        pass_last <= (mm0_r <= PASS_TH) &&
                                     ((mm_run + (last_bit_err ? 9'd1 : 9'd0)) <= PASS_TH);
                        mm_last   <= (mm_total > 10'd15) ? 4'd15 : mm_total[3:0];
                        fail_sticky <= fail_sticky |
                                       ~((mm0_r <= PASS_TH) &&
                                         ((mm_run + (last_bit_err ? 9'd1 : 9'd0)) <= PASS_TH));
                        done_ever <= 1'b1;
                        state     <= S_HOLD;
                        hold_cnt  <= 28'd0;
                    end else begin
                        cmp_ph <= 1'b1;
                    end
                end else begin
                    l0_q   <= ram_l0[cmp_idx];
                    l1_q   <= ram_l1[cmp_idx];
                    pred_q <= slot30 ? PRED_S30[cmp_idx] : PRED_S0[cmp_idx];
                    cmp_idx <= cmp_idx + 9'd1;
                    cmp_ph  <= 1'b0;
                end
            end
            // ---------------- result display, then repeat ----------
            S_HOLD: begin
                hold_cnt <= hold_cnt + 28'd1;
                if (&hold_cnt)                     // 2^28 cycles = 2.68 s
                    state <= S_START0;
            end
            default: state <= S_START0;
            endcase
        end
    end

    assign ap_start = ap_start_r;

    // ------------------------------------------------------------------
    // LEDs
    // ------------------------------------------------------------------
    reg [26:0] hb_cnt = 27'd0;
    reg        hb     = 1'b0;
    always @(posedge clk) begin
        hb_cnt <= hb_cnt + 27'd1;
        if (&hb_cnt)
            hb <= ~hb;                          // toggle every 1.34 s
    end

    assign led[7]   = hb;
    assign led[6]   = done_ever;
    assign led[5]   = pass_last;
    assign led[4]   = fail_sticky;
    assign led[3:0] = mm_last;

endmodule
